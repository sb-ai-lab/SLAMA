# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import os
import time

import logging
import sys

formatter = logging.Formatter(
    fmt='%(asctime)s %(name)s {%(module)s:%(lineno)d} %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p'
)
# set up logging to console
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console)

from contextlib import contextmanager

import pandas as pd
from pyspark.sql import SparkSession

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# load and prepare data
# TODO: put a correct path for used_cars dataset
from lightautoml.spark.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.utils import print_exec_time

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('lightautoml')]
for logger in loggers:
    logger.setLevel(logging.INFO)
    # logger.addHandler(console)


@contextmanager
def spark_session() -> SparkSession:
    spark = (
        SparkSession
        .builder
        .appName("SPARK-LAMA-app")
        .master("spark://node4.bdcl:7077")
        .config("spark.driver.host", "node4.bdcl")
        .config("spark.driver.cores", "4")
        .config("spark.driver.memory", "16g")
        .config("spark.cores.max", "16")
        .config("spark.executor.instances", "4")
        .config("spark.executor.memory", "16g")
        .config("spark.executor.cores", "4")
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.5")
        .config("spark.sql.autoBroadcastJoinThreshold", "100MB")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    # time.sleep(600)
    try:
        yield spark
    finally:
        # time.sleep(600)
        spark.stop()


# run automl
if __name__ == "__main__":
    with spark_session() as spark:
        task = SparkTask("reg")

        # data = spark.read.csv(os.path.join("file://", os.getcwd(), "../data/tiny_used_cars_data.csv"), header=True, escape="\"")
        data = spark.read.csv(os.path.join("file:///spark_data/tiny_used_cars_data.csv"), header=True, escape="\"")
        data = data.cache()
        train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

        automl = TabularAutoML(
            spark=spark,
            task=task,
            general_params={"use_algos": ["lgb", "linear_l2"]}
            # general_params={"use_algos": ["linear_l2"]}
            # general_params={"use_algos": ["lgb"]}
        )

        with print_exec_time():
            oof_predictions = automl.fit_predict(
                train_data,
                roles={"target": "price", "drop": ["dealer_zip", "description", "listed_date", "year"]}
            )
