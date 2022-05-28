import logging.config

import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from examples_utils import get_spark_session
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def main(spark: SparkSession, dataset_name: str, seed: int):
    execs = int(spark.conf.get('spark.executor.instances'))
    cores = int(spark.conf.get('spark.executor.cores'))
    memory = spark.conf.get('spark.executor.memory')
    task_type = "reg"

    dataset_increase_factor = 10

    roles = {"target": "price"}
    # test_data_dump_path = "/mnt/nfs/spark-lama-dumps/lgb_test_data.parquet"
    test_data_dump_path = "/tmp/spark_results/lgb_test_data.parquet"
    automl_model_path = "/tmp/spark_results/automl_pipeline"

    # train_data, test_data = prepare_test_and_train(spark, path, seed)
    test_data = spark.read.parquet(test_data_dump_path)
    # test_data = test_data.sample(fraction=0.0002, seed=100)

    if dataset_increase_factor >= 1:
        test_data = test_data.withColumn("new_col",
                                         F.explode(F.array(*[F.lit(0) for i in range(dataset_increase_factor)])))
        test_data = test_data.drop("new_col")
        test_data = test_data.select(
            *[c for c in test_data.columns if c != SparkDataset.ID_COLUMN],
            F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
        ).cache()
        test_data = test_data.repartition(execs * cores, SparkDataset.ID_COLUMN).cache()
        test_data = test_data.cache()
        test_data.write.mode('overwrite').format('noop').save()
        logger.info(f"Duplicated dataset size: {test_data.count()}")

    with log_exec_timer("Loading model time") as loading_timer:
        pipeline_model = PipelineModel.load(automl_model_path)

    with log_exec_timer("spark-lama predicting on test") as predict_timer:
        te_pred = pipeline_model.transform(test_data)
        te_pred = te_pred.cache()
        te_pred.write.mode('overwrite').format('noop').save()

    task = SparkTask(task_type)
    pred_column = next(c for c in te_pred.columns if c.startswith('prediction'))
    score = task.get_dataset_metric()
    test_metric_value = score(te_pred.select(
        SparkDataset.ID_COLUMN,
        F.col(roles['target']).alias('target'),
        F.col(pred_column).alias('prediction')
    ))

    logger.info(f"score for test predictions via loaded pipeline: {test_metric_value}")

    result = {
        "predict_data.count": test_data.count(),
        "spark.executor.instances": execs,
        "spark.executor.cores": cores,
        "spark.executor.memory": memory,
        "test_metric_value": test_metric_value,
        "predict_duration_secs": predict_timer.duration
    }

    print(f"EXP-RESULT: {result}")

    return result


if __name__ == "__main__":
    spark_sess = get_spark_session()
    # One can run:
    # 1. main(dataset_name="used_cars_dataset", seed=42)
    # 2. multirun(spark_sess, dataset_name="used_cars_dataset")
    main(spark_sess, dataset_name="used_cars_dataset_1x", seed=42)
    #
    # import time
    # time.sleep(600)

    spark_sess.stop()
