import logging.config
import os
from typing import Tuple, Optional, List, Union, Sequence
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F, types as SparkTypes, Window
from tqdm import tqdm

from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataFrame, SparkDataset
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

def prepare_test_and_train(spark: SparkSession, path:str, seed: int) -> Tuple[SparkDataFrame, SparkDataFrame]:
    data = spark.read.csv(path, header=True, escape="\"")

    data = data.select(
        '*',
        F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
        F.rand(seed).alias('is_test')
    ).cache()
    data.write.mode('overwrite').format('noop').save()

    train_data = data.where(F.col('is_test') < 0.8).drop('is_test').cache()
    test_data = data.where(F.col('is_test') >= 0.8).drop('is_test').cache()

    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    return train_data, test_data


def get_spark_session():
    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        return SparkSession.builder.getOrCreate()

    spark_sess = (
        SparkSession
        .builder
        .master("local[8]")
        .config("spark.jars", "/mnt/hgfs/Projects/Sber/LAMA/Sber-LAMA/jars/spark-lightautoml_2.12-0.1.jar")
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "12g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    return spark_sess


if __name__ == "__main__":
    spark = get_spark_session()

    seed = 42
    cv = 2
    use_algos = [["linear_l2"]]  # [["lgb", "linear_l2"], ["lgb"]]
    path = "/opt/spark_data/small_used_cars_data.csv"
    task_type = "reg"
    roles = {
        "target": "price",
        "drop": ["dealer_zip", "description", "listed_date",
                 "year", 'Unnamed: 0', '_c0',
                 'sp_id', 'sp_name', 'trimId',
                 'trim_name', 'major_options', 'main_picture_url',
                 'interior_color', 'exterior_color'],
        "numeric": ['latitude', 'longitude', 'mileage']
    }

    with log_exec_timer("spark-lama training") as train_timer:
        task = SparkTask(task_type)
        train_data, test_data = prepare_test_and_train(spark, path, seed)

        test_data_dropped = test_data

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            lgb_params={'use_single_dataset_mode': True},
            linear_l2_params={"default_params": {"regParam": [1]}},
            general_params={"use_algos": use_algos},
            reader_params={"cv": cv, "advanced_roles": False},
            tuning_params={'fit_on_holdout': True, 'max_tuning_iter': 50, 'max_tuning_time': 3600}
        )

        oof_predictions = automl.fit_predict(
            train_data,
            roles=roles
        )

    logger.info("Predicting on out of fold")

    score = task.get_dataset_metric()
    metric_value = score(oof_predictions)

    logger.info(f"score for out-of-fold predictions: {metric_value}")

    feature_scores = automl.get_feature_scores(calc_method="fast", data=test_data_dropped, silent=False)
    print(feature_scores)
