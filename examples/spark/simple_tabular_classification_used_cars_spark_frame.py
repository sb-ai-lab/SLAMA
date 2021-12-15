# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import os
import time

import logging
import sys

from lightautoml.spark.dataset.base import SparkDataset
from pyspark.sql import functions as F
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.transformers.categorical import LabelEncoder
from lightautoml.transformers.base import ColumnsSelector

formatter = logging.Formatter(
    fmt='%(asctime)s %(name)s {%(module)s:%(lineno)d} %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p'
)

logging.basicConfig(level=logging.INFO)
# set up logging to console
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)

# root_logger = logging.getLogger('')
root_logger = logging.getLogger('lightautoml.spark')
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console)

# root_logger = logging.getLogger('')
root_logger = logging.getLogger('lightautoml')
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console)


# root_logger = logging.getLogger('')
root_logger = logging.getLogger('lightautoml.ml_algo')
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

# loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('lightautoml')]
#
# for logger in loggers:
#     logger.setLevel(logging.DEBUG)
#     # logger.addHandler(console)

@contextmanager
def spark_session() -> SparkSession:
    spark = (
        SparkSession
        .builder
        .appName("SPARK-LAMA-app")
        # .master("local[4]")
        .master("spark://node4.bdcl:7077")
        .config("spark.driver.host", "node4.bdcl")
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4")
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
        .config("spark.driver.cores", "4")
        .config("spark.driver.memory", "16g")
        .config("spark.cores.max", "16")
        .config("spark.executor.instances", "4")
        .config("spark.executor.memory", "16g")
        .config("spark.executor.cores", "4")
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.5")
        .config("spark.sql.autoBroadcastJoinThreshold", "100MB")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    # time.sleep(600)
    try:
        yield spark
    finally:
        time.sleep(600)
        spark.stop()


# run automl
if __name__ == "__main__":
    with spark_session() as spark:
        # data = spark.read.csv(os.path.join("file:///opt/0125l_dataset.csv"), header=True, escape="\"")
        #
        # sreader = SparkToSparkReader(task=SparkTask("reg"), cv=1)
        # sds = sreader.fit_read(data, roles={
        #     "target": "price",
        #     "drop": ["dealer_zip", "description", "listed_date", "year"],
        #     "numeric": ['latitude', 'longitude', 'mileage']
        # })
        #
        # colsel = ColumnsSelector(keys=get_columns_by_role(sds, "Category"))
        # enc_sds = colsel.fit_transform(sds)
        #
        # enc = LabelEncoder()
        # enc_sds = enc.fit_transform(enc_sds)
        # enc_sds.cache()
        #
        # import pprint
        # from pyspark.sql import functions as F
        # num_cols = get_columns_by_role(sds, "Numeric")
        # row = sds.data.select(*[F.sum(F.isnull(c).astype('int')).alias(c) for c in num_cols]).first()
        # pprint.pprint(row.asDict())
        #
        # raise NotImplementedError()

        task = SparkTask("reg")

        # data = spark.read.csv(os.path.join("file://", os.getcwd(), "../data/tiny_used_cars_data.csv"), header=True, escape="\"")
        # data = spark.read.csv(os.path.join("file:///spark_data/tiny_used_cars_data.csv"), header=True, escape="\"")
        # data = spark.read.csv(os.path.join("file:///spark_data/derivative_datasets/0125x_cleaned.csv"), header=True, escape="\"")
        # data = spark.read.csv(os.path.join("file:///spark_data/derivative_datasets/4x_cleaned.csv"), header=True, escape="\"")
        data = spark.read.csv(os.path.join("file:///spark_data/derivative_datasets/2x_cleaned.csv"), header=True, escape="\"")
        # data = spark.read.csv(os.path.join("file:///opt/0125l_dataset.csv"), header=True, escape="\"")
        data = data.cache()
        train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

        test_data_dropped = test_data \
                                .withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id()) \
                                .drop(F.col("price")).cache()

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

        te_pred = automl.predict(test_data_dropped)

        test_data_pd = test_data.toPandas()
        te_pred_pd = te_pred.data.limit(50).toPandas()

        _a = 1  # Just to check results in the debuggerr
