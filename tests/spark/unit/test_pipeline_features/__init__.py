import os
from typing import Tuple, get_args, cast, List, Optional, Dict

import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset, NumpyDataset
from lightautoml.dataset.roles import ColumnRole
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.transformers.base import SparkTransformer
from lightautoml.transformers.base import LAMLTransformer
from lightautoml.transformers.numeric import NumpyTransformable

import numpy as np
import pandas as pd

# NOTE!!!
# All tests require PYSPARK_PYTHON env variable to be set
# for example: PYSPARK_PYTHON=/home/nikolay/.conda/envs/LAMA/bin/python


@pytest.fixture(scope="session")
def spark() -> SparkSession:

    spark = SparkSession.builder.config("master", "local[1]").getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


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


def compare_obtained_datasets(lama_ds: NumpyDataset, spark_ds: SparkDataset):
    lama_np_ds = cast(NumpyTransformable, lama_ds).to_numpy()
    spark_np_ds = spark_ds.to_numpy()

    assert list(sorted(lama_np_ds.features)) == list(sorted(spark_np_ds.features)), \
        f"List of features are not equal\n" \
        f"LAMA: {sorted(lama_np_ds.features)}\n" \
        f"SPARK: {sorted(spark_np_ds.features)}"

    # compare roles equality for the columns
    assert lama_np_ds.roles == spark_np_ds.roles, "Roles are not equal"

    # compare shapes
    assert lama_np_ds.shape == spark_np_ds.shape, "Shapes are not equals"

    lama_data: np.ndarray = lama_np_ds.data
    spark_data: np.ndarray = spark_np_ds.data
    features: List[int] = [i for i, _ in sorted(enumerate(lama_np_ds.features), key=lambda x: x[1])]

    assert np.allclose(
        np.sort(lama_data[:, features], axis=0), np.sort(spark_data[:, features], axis=0),
        equal_nan=True
    ), \
        f"Results of the LAMA's transformer and the Spark based transformer are not equal: " \
        f"\n\nLAMA: \n{lama_data}" \
        f"\n\nSpark: \n{spark_data}"