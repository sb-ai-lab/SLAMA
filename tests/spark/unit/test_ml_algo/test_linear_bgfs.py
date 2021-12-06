import pickle

import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.linear_pyspark import LinearLBFGS
from lightautoml.tasks import Task
from lightautoml.validation.base import DummyIterator
from .. import from_pandas_to_spark, spark

import pandas as pd


def test_smoke_linear_bgfs(spark: SparkSession):
    with open("unit/resources/datasets/dump_tabular_automl_lgb_cb_linear/Lvl_0_Pipe_0_apply_selector.pickle", "rb") as f:
        data, target, features, roles = pickle.load(f)

    nds = NumpyDataset(data, features, roles, task=Task("binary"))
    pds = nds.to_pandas()
    target = pd.Series(target)

    sds = from_pandas_to_spark(pds, spark, target)
    iterator = DummyIterator(train=sds)

    ml_algo = LinearLBFGS()
    pred_ds = ml_algo.fit_predict(iterator)

    predicted_sdf = pred_ds.data
    predicted_sdf.show(10)

    assert SparkDataset.ID_COLUMN in predicted_sdf.columns
    assert len(pred_ds.features) == 1
    assert pred_ds.features[0].endswith("_prediction")
    assert pred_ds.features[0] in predicted_sdf.columns
