# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging.config
import os

from pyspark.sql import functions as F

from lightautoml.spark.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.utils import log_exec_time, logging_config, VERBOSE_LOGGING_FORMAT, spark_session

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


# run automl
if __name__ == "__main__":
    with spark_session(master="local[1]") as spark:
        task = SparkTask("reg")
        data = spark.read.csv(os.path.join("file://", os.getcwd(), "../data/tiny_used_cars_data.csv"),
                              header=True, escape="\"")
        # data = spark.read.csv("file:///spark_data/tiny_used_cars_data.csv", header=True, escape="\"")
        # data = spark.read.csv("file:///spark_data/derivative_datasets/0125x_cleaned.csv", header=True, escape="\"")
        # data = spark.read.csv("file:///spark_data/derivative_datasets/4x_cleaned.csv", header=True, escape="\"")
        # data = spark.read.csv("file:///spark_data/derivative_datasets/2x_cleaned.csv", header=True, escape="\"")
        # data = spark.read.csv("file:///opt/0125l_dataset.csv", header=True, escape="\"")
        data = data.cache()
        train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

        test_data_dropped = test_data\
            .withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id())\
            .drop(F.col("price")).cache()

        automl = TabularAutoML(spark=spark, task=task, general_params={"use_algos": ["lgb", "linear_l2"]})

        with log_exec_time():
            oof_predictions = automl.fit_predict(
                train_data,
                roles={
                    "target": "price",
                    "drop": ["dealer_zip", "description", "listed_date",
                             "year", 'Unnamed: 0', '_c0',
                             'sp_id', 'sp_name', 'trimId',
                             'trim_name', 'major_options', 'main_picture_url',
                             'interior_color', 'exterior_color'],
                    "numeric": ['latitude', 'longitude', 'mileage']
                }
            )

        logger.info("Predicting on out of fold")

        # TODO: SPARK-LAMA fix bug in SparkToSparkReader.read method
        # with log_exec_time():
        #     te_pred = automl.predict(test_data_dropped)
        #     test_data_pd = test_data.toPandas()
        #     te_pred_pd = te_pred.data.limit(50).toPandas()
