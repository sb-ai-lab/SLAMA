import logging.config

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from examples_utils import BUCKET_NUMS, BASE_HDFS_PREFIX
from examples_utils import get_spark_session
from sparklightautoml.automl.presets.tabular_presets import SparkTabularAutoML
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import VERBOSE_LOGGING_FORMAT
from sparklightautoml.utils import log_exec_timer
from sparklightautoml.utils import logging_config

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename="/tmp/slama.log"))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def main(spark: SparkSession, seed: int):
    use_algos = [["lgb"]]
    cv = 5
    task_type = "reg"

    persistence_manager = PlainCachePersistenceManager()

    with log_exec_timer("spark-lama training") as train_timer:
        task = SparkTask(task_type)

        target = 'price'
        train_data = spark.read.parquet(
            # f"{BASE_HDFS_PREFIX}/opt/preprocessed_datasets/adv_small_used_cars_dataset.slama/data.parquet"
            # f"{BASE_HDFS_PREFIX}/opt/preprocessed_datasets/adv_used_cars_10x.parquet"
            f"{BASE_HDFS_PREFIX}/opt/preprocessed_datasets/adv_used_cars_100x.parquet"
        )
        numeric_cols = [c for c in train_data.columns if c not in ['_id', 'reader_fold_num', target]]
        numeric_cols = {c: c.replace('[', '(').replace(']', ')') for c in numeric_cols}
        train_data = train_data.select(target, *(sf.col(c).alias(c_val) for c, c_val in numeric_cols.items()))
        roles = {
            "target": target,
            "numeric": list(numeric_cols.values())
        }

        # optionally: set 'convert_to_onnx': True to use onnx-based version of lgb's model transformer
        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            general_params={"use_algos": use_algos},
            selection_params={"select_algos": []},
            # execution mode only available for synapseml 0.11.1
            lgb_params={
                "default_params": {
                  "numIterations": 50,
                },
                "use_single_dataset_mode": True,
                "execution_mode": "streaming",
                "convert_to_onnx": False,
                "mini_batch_size": 1000,
                "freeze_defaults": True
            },
            linear_l2_params={"default_params": {"regParam": [1e-5]}},
            reader_params={"cv": cv, "advanced_roles": False},
        )

        oof_predictions = automl.fit_predict(train_data, roles=roles, persistence_manager=persistence_manager)

    logger.info("Predicting on out of fold")

    score = task.get_dataset_metric()
    metric_value = score(oof_predictions)

    logger.info(f"score for out-of-fold predictions: {metric_value}")

    oof_predictions.unpersist()
    # this is necessary if persistence_manager is of CompositeManager type
    # it may not be possible to obtain oof_predictions (predictions from fit_predict) after calling unpersist_all
    automl.persistence_manager.unpersist_all()

    result = {
        "seed": seed,
        "used_algo": str(use_algos),
        "metric_value": metric_value,
        "train_duration_secs": train_timer.duration,
    }

    print(f"EXP-RESULT: {result}")

    return result


if __name__ == "__main__":
    spark_sess = get_spark_session(BUCKET_NUMS)
    main(spark_sess, seed=42)

    spark_sess.stop()
