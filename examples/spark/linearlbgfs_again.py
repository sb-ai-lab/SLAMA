# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import pickle
import time
from datetime import datetime
import logging
import os
import sys
from typing import cast

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from skimage.metrics import mean_squared_error
from pyspark.ml.regression import LinearRegression
from synapse.ml.lightgbm import LightGBMRegressor, LightGBMClassifier

from pyspark.sql import functions as F

from lightautoml.dataset.np_pd_dataset import NumpyDataset

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
logger = root_logger

from contextlib import contextmanager

from pyspark.sql import SparkSession


loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('lightautoml')]
for logger in loggers:
    logger.setLevel(logging.INFO)
    # logger.addHandler(console)


@contextmanager
def print_exec_time(name: str = None):
    start = datetime.now()
    yield
    end = datetime.now()
    duration = (end - start).total_seconds()
    print(f"Exec time ({name}): {duration}")


spark = (
    SparkSession
        .builder
        .appName("SPARK-LAMA-app")
        .master("local[4]")
        # .master("spark://node4.bdcl:7077")
        # .config("spark.driver.host", "node4.bdcl")
        .config("spark.driver.cores", "4")
        .config("spark.driver.memory", "16g")
        .config("spark.cores.max", "8")
        .config("spark.executor.instances", "2")
        .config("spark.executor.memory", "16g")
        .config("spark.executor.cores", "4")
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.5")
        .config("spark.sql.autoBroadcastJoinThreshold", "100MB")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
)
@contextmanager
def spark_session() -> SparkSession:

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

        # /mnt/ess_storage/DN_1/storage/sber_LAMA/kaggle_used_cars_dataset
        # base_path = '/spark_data/dumps_05x'
        base_path = '/opt'

        logger.info("Start to read data from local file")
        with open(os.path.join(base_path, 'lama_train_lgb'), 'rb') as f:
            data = pickle.load(f)
            data = cast(NumpyDataset, data)

        logger.info("Converting it to pandas")
        train_data_pdf = data.to_pandas()

        logger.info("Converting from pandas to Spark frame")
        train_data_pdf.data['price'] = train_data_pdf.target
        temp_data = train_data_pdf.data
        train_data = spark.createDataFrame(temp_data).drop('Unnamed: 0')
        cols = train_data.columns
        train_data = (
            train_data
            .withColumn("dummy", F.explode(F.array(*[F.lit(i) for i in range(1)])))
            .select(cols)
            .repartition(200)
            .cache()
        )

        logger.info("Starting to train")

        with open(os.path.join(base_path, 'lama_valid_lgb'), 'rb') as f:
            data = pickle.load(f)
            data = cast(NumpyDataset, data)

        valid_data_pdf = data.to_pandas()
        valid_data_pdf.data['price'] = valid_data_pdf.target
        temp_data = valid_data_pdf.data
        valid_data = spark.createDataFrame(temp_data).drop('Unnamed: 0').cache()

        is_reg = True
        name = "linear_l2"
        LGBMBooster = LightGBMRegressor if is_reg else LightGBMClassifier

        # categorical features
        cat_feats = [feat for feat in train_data_pdf.features if feat not in ['Unnamed: 0', 'price'] and train_data_pdf.roles[feat].name == "Category"]
        non_cat_feats = [feat for feat in train_data_pdf.features if feat not in ['Unnamed: 0', 'price'] and train_data_pdf.roles[feat].name != "Category"]

        ohe = OneHotEncoder(inputCols=cat_feats, outputCols=[f"{f}_{name}_ohe" for f in cat_feats])
        assembler = VectorAssembler(
            inputCols=non_cat_feats + ohe.getOutputCols(),
            outputCol=f"{name}_vassembler_features",
            handleInvalid='keep'
        )

        # TODO: SPARK-LAMA add params processing later
        if not is_reg:
            model = LogisticRegression(featuresCol=assembler.getOutputCol(),
                                       labelCol='price',
                                       predictionCol='predict')
            # **params)
        else:
            model = LinearRegression(
                featuresCol=assembler.getOutputCol(),
                labelCol='price',
                predictionCol='predict',
                maxIter=100,
                tol=1e-6,
                loss="squaredError"
            )
            # **params)
            model.setSolver("l-bfgs")

        pipeline = Pipeline(stages=[ohe, assembler, model])

        lgbm = pipeline

        with print_exec_time("training"):
            ml_model = lgbm.fit(train_data)

        with print_exec_time("train prediction"):
            train_pred = ml_model.transform(train_data)
            train_pred_pdf = train_pred.select('price', 'predict').toPandas()

        with print_exec_time("val prediction"):
            val_pred = ml_model.transform(valid_data)
            val_pred_pdf = val_pred.select('price', 'predict').toPandas()

        print(f"Score for train:{mean_squared_error(train_pred_pdf['predict'].values, train_pred_pdf['price'].values)}")
        print(f"Score for hold-out:{mean_squared_error(val_pred_pdf['predict'].values, val_pred_pdf['price'].values)}")

        # time.sleep(600)
