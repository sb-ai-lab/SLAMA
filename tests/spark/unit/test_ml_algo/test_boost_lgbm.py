import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.spark.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.spark.validation.base import DummyIterator as SparkDummyIterator
from . import DatasetForTest, spark
from ..test_transformers import from_pandas_to_spark

DATASETS = [

    # DatasetForTest("test_transformers/resources/datasets/dataset_23_cmc.csv", default_role=CategoryRole(np.int32)),

    DatasetForTest("test_transformers/resources/datasets/house_prices.csv",
                   columns=["Id",
                            "MSSubClass",
                            # "MSZoning",
                            "LotFrontage"],
                   roles={
                       "Id": CategoryRole(np.int32),
                       "MSSubClass": CategoryRole(np.int32),
                       # "MSZoning": CategoryRole(str),
                       "LotFrontage": CategoryRole(np.float32)
                   })
]


@pytest.mark.parametrize("dataset", DATASETS)
def test_smoke_boost_lgbm(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    iterator = SparkDummyIterator(train=from_pandas_to_spark(ds, spark))

    lgbm = BoostLGBM()

    predicted = lgbm.fit_predict(iterator).data

    predicted.show(10)




