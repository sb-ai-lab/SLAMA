import logging
import logging.config
import pickle
import sys

import sklearn
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import BooleanType
from synapse.ml.lightgbm import LightGBMClassifier

from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.utils import spark_session, logging_config, VERBOSE_LOGGING_FORMAT
from lightautoml.validation.base import DummyIterator, HoldoutIterator
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM as SparkBoostLGBM
from lightautoml.spark.ml_algo.linear_pyspark import LinearLBFGS as SparkLinearLBFGS

from pyspark.sql import functions as F

import numpy as np

import pandas as pd

# TODO: need to log data in predict
# TODO: correct order in PandasDataset from Spark ?
# TODO: correct parameters of BoostLGBM?
# TODO: correct parametes of Tuner for BoostLGBM?
from tests.spark.unit import from_pandas_to_spark

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

mode = "spark"

# task_name, target_col = "used_cars", "price"
task_name, target_col = "binary", "TARGET"

# alg = "linear_l2"
alg = "lgb"

path = f'../dumps/datalog_{task_name}_{mode}_{alg}_train_val.pickle'
path_for_test = f'../dumps/datalog_{task_name}_{mode}_{alg}_test_part.pickle'
path_predict = f'../dumps/datalog_{task_name}_{mode}_{alg}_predict.pickle'


with open(path, "rb") as f:
    data = pickle.load(f)
    train = data['data']['train']
    valid = data['data']['valid']

with open(path_for_test, "rb") as f:
    test_target_df = pickle.load(f)
    test_target_df = test_target_df['data']['test']

#     # if mode == "spark":
#     #     test_target_df.sort_values(SparkDataset.ID_COLUMN, inplace=True)

with open(path_predict, "rb") as f:
    test_df = pickle.load(f)
    test_df = test_df['data']['predict']

# TODO: verification by _id equality

# tgts = test_df.target

if mode == "spark":
    tgts = test_df.target
else:
    tgts = test_target_df[target_col].values


train_lama = train if alg == "lgb" else train.to_numpy()
valid_lama = valid if alg == "lgb" else valid.to_numpy()
test_df_lama = test_df if alg == "lgb" else test_df.to_numpy()

# train_valid = DummyIterator(train_lama) if alg == "lgb" else DummyIterator(train_lama)
train_valid = HoldoutIterator(train_lama, valid_lama)
ml_algo = BoostLGBM() if alg == "lgb" else LinearLBFGS()

ml_algo, _ = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)

preds = ml_algo.predict(test_df_lama)

if train.task.name == "binary":
    evaluator = sklearn.metrics.roc_auc_score
elif train.task.name == "reg":
    evaluator = sklearn.metrics.mean_squared_error
else:
    raise ValueError()

test_metric_value = evaluator(tgts, preds.data[:, 0])

print(f"Test metric value: {test_metric_value}")

# sys.exit(1)
# =================================================
# if mode == "spark":
if True:
    with spark_session('local[4]') as spark:
        train_sds = from_pandas_to_spark(train.to_pandas(), spark, pd.Series(train.target))
        valid_sds = from_pandas_to_spark(valid.to_pandas(), spark, pd.Series(valid.target))
        test_sds = from_pandas_to_spark(test_df.to_pandas(), spark, pd.Series(tgts))
        iterator = HoldoutIterator(train_sds, valid_sds)

        #
        # predict_col = "prediction"
        # cat_feats = [feat for feat in train_sds.features if train_sds.roles[feat].name == "Category"]
        # non_cat_feats = [feat for feat in train_sds.features if train_sds.roles[feat].name != "Category"]
        #
        # ohe = OneHotEncoder(inputCols=cat_feats, outputCols=[f"{f}_lgbfs_ohe" for f in cat_feats])
        # assembler = VectorAssembler(
        #     inputCols=non_cat_feats + ohe.getOutputCols(),
        #     outputCol=f"lgbfs_vassembler_features"
        # )
        #
        # # TODO: SPARK-LAMA add params processing later
        # # model = LogisticRegression(featuresCol=assembler.getOutputCol(),
        # #                            labelCol=train_sds.target_column,
        # #                            predictionCol=predict_col)
        # model = LinearRegression(featuresCol=assembler.getOutputCol(),
        #                          labelCol=train_sds.target_column,
        #                          predictionCol=predict_col)
        # # **params)
        # model.setSolver("l-bfgs")
        #
        # pipeline = Pipeline(stages=[ohe, assembler, model])
        #
        # train_df = train_sds.data\
        #     .join(train_sds.target, on=SparkDataset.ID_COLUMN)\
        #     .withColumn("tr_or_val", F.floor(F.rand(42) / 0.8).cast(BooleanType()))\
        #     .cache()
        # # test = test_sds.data.join(test_sds.target, on=SparkDataset.ID_COLUMN).cache()
        # # train_df, valid_df = train_df.randomSplit([0.8, 0.2], seed=42)
        # # train_df, valid_df = train_df.cache(), valid_df.cache()
        # # train_df.count()
        # # valid_df.count()
        #
        # model = pipeline.fit(train_df)
        # preds_df = model.transform(train_sds.data)
        #
        # pred_target_df = (
        #     preds_df
        #     .join(test_sds.target, on=SparkDataset.ID_COLUMN, how='inner')
        #     .select(SparkDataset.ID_COLUMN, test_sds.target_column, predict_col)
        # )

        # ### Normal way
        spark_ml_algo = SparkBoostLGBM() if alg == "lgb" else SparkLinearLBFGS()
        spark_ml_algo, _ = tune_and_fit_predict(spark_ml_algo, DefaultTuner(), iterator)
        preds = spark_ml_algo.predict(test_sds)
        predict_col = preds.features[0]
        preds = preds.data

        pred_target_df = preds.select(SparkDataset.ID_COLUMN, test_sds.target_column, predict_col)

        pt_df = pred_target_df.toPandas()
        test_metric_value2 = evaluator(
            pt_df[train_sds.target_column].values,
            pt_df[predict_col].values
        )

        print(f"Test metric value2: {test_metric_value2}")
