from typing import cast

from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.spark.dataset import SparkDataset

import pandas as pd


def from_pandas_to_spark(p: PandasDataset, spark: SparkSession) -> SparkDataset:
    pdf = cast(pd.DataFrame, p.data)
    sdf = spark.createDataFrame(data=pdf)
    return SparkDataset(sdf, roles=p.roles)