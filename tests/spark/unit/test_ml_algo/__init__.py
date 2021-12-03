from typing import Tuple, get_args, cast, List, Optional, Dict

import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset, NumpyDataset
from lightautoml.dataset.roles import ColumnRole
from lightautoml.spark.transformers.base import SparkTransformer
from lightautoml.spark.utils import from_pandas_to_spark
from lightautoml.transformers.base import LAMLTransformer
from lightautoml.transformers.numeric import NumpyTransformable

import numpy as np
import pandas as pd

# NOTE!!!
# All tests require PYSPARK_PYTHON env variable to be set
# for example: PYSPARK_PYTHON=/home/nikolay/.conda/envs/LAMA/bin/python


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = SparkSession.builder.config("master", "local[1]") \
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


def smoke_check(spark: SparkSession, ds: PandasDataset, t_spark: SparkTransformer) -> NumpyDataset:
    sds = from_pandas_to_spark(ds, spark)

    t_spark.fit(sds)
    transformed_sds = t_spark.transform(sds)

    spark_np_ds = transformed_sds.to_numpy()

    return spark_np_ds


class DatasetForTest:
    def __init__(self, path: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None,
                 columns: Optional[List[str]] = None,
                 roles: Optional[Dict] = None,
                 default_role: Optional[ColumnRole] = None):

        if path is not None:
            self.dataset = pd.read_csv(path)
        else:
            self.dataset = df

        if columns is not None:
            self.dataset = self.dataset[columns]

        if roles is None:
            self.roles = {name: default_role for name in self.dataset.columns}
        else:
            self.roles = roles
