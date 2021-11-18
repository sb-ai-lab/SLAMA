import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.spark.transformers.decomposition import PCATransformer as SparkPCATransformer
from lightautoml.transformers.decomposition import PCATransformer
from . import compare_by_metadata, spark


def test_pca(spark: SparkSession):
    source_data = pd.DataFrame(data={
        "a": [0.1, 34.7, 21.34, 2.01, 5.0],
        "b": [0.12, 1.7, 28.38, 0.002, 1.4],
        "c": [0.11, 12.67, 89.1, 56.1, -0.99],
        "d": [0.001, 0.003, 0.5, 0.991, 0.1]
    })

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})

    # we are doing here 'smoke test' to ensure that it even can run at all
    # and also a check for metadat validity: features, roles, shapes should be ok
    lama_ds, spark_ds = compare_by_metadata(
        spark, ds, PCATransformer(n_components=10), SparkPCATransformer(n_components=10)
    )

    spark_data: np.ndarray = spark_ds.data

    # doing minor content check
    assert all(spark_data.flatten()), f"Data should not contain None-s: {spark_data.flatten()}"
