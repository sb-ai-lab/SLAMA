import numpy as np
import pandas as pd
import pytest
import pickle
from pyspark.sql import SparkSession

from lightautoml.tasks.base import Task
from lightautoml.validation.base import DummyIterator
from lightautoml.dataset.np_pd_dataset import NumpyDataset
from lightautoml.spark.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.spark.dataset.base import SparkDataset

from . import spark
from ..test_transformers import from_pandas_to_spark


def test_smoke_boost_lgbm_v2(spark: SparkSession):

    with open("unit/test_ml_algo/datasets/Lvl_0_Pipe_0_apply_selector.pickle", "rb") as f:
        data, target, features, roles = pickle.load(f)

    nds = NumpyDataset(data[4000:, :], features, roles, task=Task("binary"))
    pds = nds.to_pandas()
    target = pd.Series(target[4000:])

    sds = from_pandas_to_spark(pds, spark, target)
    iterator = DummyIterator(train=sds)

    ml_algo = BoostLGBM()
    pred_ds = ml_algo.fit_predict(iterator)

    predicted_sdf = pred_ds.data
    ppdf = predicted_sdf.toPandas()

    assert SparkDataset.ID_COLUMN in predicted_sdf.columns
    assert len(pred_ds.features) == 1
    assert pred_ds.features[0].endswith("_prediction")
    assert pred_ds.features[0] in predicted_sdf.columns


