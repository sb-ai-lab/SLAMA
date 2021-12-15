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

from pyspark.ml.feature import VectorAssembler
# from skimage.metrics import mean_squared_error
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
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4")
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
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
        # base_path = '/opt'
        path = '/tmp/dump_before_lgbm.parquet'

        logger.info("Start to read data from local file")
        # path = os.path.join(base_path, 'lama_train_lgb') if os.path.isdir(base_path) else base_path
        # with open(path, 'rb') as f:
        #     data = pickle.load(f)
        #     data = cast(NumpyDataset, data)
        #
        # logger.info("Converting it to pandas")
        # train_data_pdf = data.to_pandas()

        # logger.info("Converting from pandas to Spark frame")
        # train_data_pdf.data['price'] = train_data_pdf.target
        # temp_data = train_data_pdf.data
        # train_data = spark.createDataFrame(temp_data).drop('Unnamed: 0')
        # features = train_data_pdf.features

        # train_data = spark.read.parquet(path)

        import pandas as pd
        import random
        with open("/opt/dump_selector_lgbm_0125l.pickle", "rb") as f:
            train_data = pickle.load(f)
            train_data['price'] = pd.Series([ random.randint(0, 1) for _ in range(train_data.shape[0])])
            train_data = spark.createDataFrame(train_data)

        train_data = train_data.select(*[F.when(F.isnull(c), float('nan')).otherwise(F.col(c).astype('float')).alias(c) for c in train_data.columns])
        train_data = train_data.cache()

        features = train_data.columns

        cols = train_data.columns
        train_data = (
            train_data
            .withColumn("dummy", F.explode(F.array(*[F.lit(i) for i in range(1)])))
            .select(cols)
            .repartition(200)
            .cache()
        )

        logger.info("Starting to train")

        # with open(os.path.join(base_path, 'lama_valid_lgb'), 'rb') as f:
        #     data = pickle.load(f)
        #     data = cast(NumpyDataset, data)
        #
        # valid_data_pdf = data.to_pandas()
        # valid_data_pdf.data['price'] = valid_data_pdf.target
        # temp_data = valid_data_pdf.data
        # valid_data = spark.createDataFrame(temp_data).drop('Unnamed: 0').cache()

        is_reg = True
        LGBMBooster = LightGBMRegressor if is_reg else LightGBMClassifier

        assembler = VectorAssembler(
            inputCols=[f for f in features if f != 'Unnamed: 0'],
            outputCol="LightGBM_vassembler_features",
            handleInvalid="keep"
        )

        # the parameters taken from LAMA's BoostLGBM,
        # no corresponding option found for num_trees
        lgbm = LGBMBooster(
            # fobj=fobj,  # TODO SPARK-LAMA: Commented only for smoke test
            # feval=feval,
            featuresCol="LightGBM_vassembler_features",
            labelCol="price",
            predictionCol="predict",
            learningRate=0.05,
            numLeaves=128,
            featureFraction=0.9,
            baggingFraction=0.9,
            baggingFreq=1,
            maxDepth=-1,
            verbosity=-1,
            minGainToSplit=0.0,
            numThreads=1,
            maxBin=255,
            minDataInLeaf=3,
            earlyStoppingRound=100,
            metric="mse",
            numIterations=10
        )

        if is_reg:
            lgbm.setAlpha(1.0).setLambdaL1(0.0).setLambdaL2(0.0)

        with print_exec_time("training"):
            temp_sdf = assembler.transform(train_data)
            ml_model = lgbm.fit(temp_sdf)

        with print_exec_time("train prediction"):
            train_pred = ml_model.transform(temp_sdf)
            train_pred_pdf = train_pred.select('price', 'predict').toPandas()
        #
        # with print_exec_time("val prediction"):
        #     temp_sdf = assembler.transform(valid_data)
        #     val_pred = ml_model.transform(temp_sdf)
        #     val_pred_pdf = val_pred.select('price', 'predict').toPandas()

        # print(f"Score for train:{mean_squared_error(train_pred_pdf['predict'].values, train_pred_pdf['price'].values)}")
        # print(f"Score for hold-out:{mean_squared_error(val_pred_pdf['predict'].values, val_pred_pdf['price'].values)}")

        time.sleep(600)
        # print(f"Score for out-of-fold predictions: {roc_auc_score(train_data['TARGET'].values, oof_predictions.data[:, 0])}")
        # print(f"Score for hold-out: {roc_auc_score(test_data['TARGET'].values, te_pred.data[:, 0])}")
