import logging.config

from pyspark.sql import SparkSession

from examples_utils import get_persistence_manager, BUCKET_NUMS, Dataset
from examples_utils import prepare_test_and_train, get_spark_session
from sparklightautoml.automl.presets.tabular_presets import SparkTabularAutoML
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


def main(spark: SparkSession, seed: int):
    """
        This example computes an AutoML pipeline for recommendations reranking.
        (SLAMA serves as a second layer in a two-stage scenario of RePlay library).
        Algos and layers to be used during automl:

        For example:
        1. use_algos = [["lgb"]]
        2. use_algos = [["lgb_tuned"]]
        3. use_algos = [["linear_l2"]]
        4. use_algos = [["lgb", "linear_l2"], ["lgb"]]

        Alternative ways to define persistence_manager:
        1. persistence_manager = get_persistence_manager("CompositePlainCachePersistenceManager")
        2. persistence_manager = CompositePlainCachePersistenceManager(bucket_nums=BUCKET_NUMS)
    Args:
        spark: spark session to go with
        seed: random seed
    """

    use_algos = [["lgb"]]
    cv = 3
    dataset = Dataset(
        path='/opt/experiments/test_exp/full_second_level_train.parquet',
        task_type='binary',
        roles={"target": "target"},
        file_format='parquet',
        file_format_options={}
    )

    persistence_manager = get_persistence_manager()

    with log_exec_timer("spark-lama training"):
        task = SparkTask(dataset.task_type)
        train_data, test_data = prepare_test_and_train(dataset, seed)

        train_data = train_data.drop("user_idx", "item_idx")
        test_data_dropped = test_data

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            general_params={"use_algos": use_algos},
            lgb_params={
                'use_single_dataset_mode': True,
                'convert_to_onnx': False,
                'mini_batch_size': 1000
            },
            linear_l2_params={'default_params': {'regParam': [1e-5]}},
            reader_params={"cv": cv, "advanced_roles": False}
        )

        oof_predictions = automl.fit_predict(
            train_data,
            roles=dataset.roles,
            persistence_manager=persistence_manager
        )

    logger.info("Predicting on out of fold")

    score = task.get_dataset_metric()
    metric_value = score(oof_predictions)

    logger.info(f"score for out-of-fold predictions: {metric_value}")

    oof_predictions.unpersist()
    # this is necessary if persistence_manager is of CompositeManager type
    # it may not be possible to obtain oof_predictions (predictions from fit_predict) after calling unpersist_all
    automl.persistence_manager.unpersist_all()

    with log_exec_timer("spark-lama predicting on test (#2 way)"):
        te_pred = automl.transformer().transform(test_data_dropped)

        score = task.get_dataset_metric()
        test_metric_value = score(te_pred)

        logger.info(f"score for test predictions: {test_metric_value}")

    logger.info("Predicting is finished")

    train_data.unpersist()
    test_data.unpersist()


if __name__ == "__main__":
    # if one uses bucketing based persistence manager,
    # the argument below number should be equal to what is set to 'bucket_nums' of the manager
    spark_sess = get_spark_session(BUCKET_NUMS)
    main(spark_sess, seed=42)

    spark_sess.stop()
