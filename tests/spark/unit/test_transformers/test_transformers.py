import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import NumericRole, TextRole, CategoryRole
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.transformers.decomposition import PCATransformer as SparkPCATransformer
from lightautoml.spark.transformers.numeric import NaNFlags as SparkNaNFlags, \
    FillnaMedian as SparkFillnaMedian, LogOdds as SparkLogOdds, StandardScaler as SparkStandardScaler, \
    QuantileBinning as SparkQuantileBinning
from lightautoml.spark.transformers.categorical import OHEEncoder as SparkOHEEncoder
from lightautoml.spark.transformers.text import TfidfTextTransformer as SparkTfidfTextTransformer
from lightautoml.transformers.categorical import OHEEncoder
from lightautoml.transformers.decomposition import PCATransformer
from lightautoml.transformers.numeric import NaNFlags, FillnaMedian, LogOdds, StandardScaler, QuantileBinning
from . import compare_by_content, compare_by_metadata, smoke_check


# Note:
# -s means no stdout capturing thus allowing one to see what happens in reality

# IMPORTANT !
# The test requires env variable PYSPARK_PYTHON to be set
# for example: PYSPARK_PYTHON=/home/<user>/.conda/envs/LAMA/bin/python


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = SparkSession.builder.config("master", "local[1]").getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


@pytest.mark.skip
def test_nan_flags(spark: SparkSession):
    nan_rate = 0.2
    source_data = pd.DataFrame(data={
        "a": [None if i >= 5 else i for i in range(10)],
        "b": [None if i >= 7 else i for i in range(10)],
        "c": [None if i == 2 else i for i in range(10)],
        "d": list(range(10))
    })

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})

    compare_by_content(spark, ds, NaNFlags(nan_rate), SparkNaNFlags(nan_rate))


@pytest.mark.skip
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


@pytest.mark.skip
def test_tfidf_text_transformer(spark: SparkSession):
    param_defaults = {
        "min_df": 1.0,
        "max_df": 100.0,
        "max_features": 15
    }
    source_data = pd.DataFrame(data={
        "a": ["ipsen loren doloren" for _ in range(10)],
        "b": ["ipsen loren doloren" for _ in range(10)],
        "c": ["ipsen loren doloren" for _ in range(10)],
    })

    ds = PandasDataset(source_data, roles={name: TextRole() for name in source_data.columns})

    # we cannot compare by content because the formulas used by Spark and scikit is slightly different
    # see: https://github.com/scikit-learn/scikit-learn/blob/0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/feature_extraction/text.py#L1461
    # and: https://spark.apache.org/docs/latest/ml-features#tf-idf
    # we also cannot compare by metadata cause the number of resulting columns will be different
    # because of Spark and sklearn treats 'max_features' param for vocab size differently

    result_ds = smoke_check(spark, ds, SparkTfidfTextTransformer(param_defaults))

    new_cols = {f.split('__')[1] for f in result_ds.features}
    assert len(result_ds.features) == len(source_data.columns)
    assert len(new_cols) == len(source_data.columns)
    assert all(isinstance(r, NumericVectorOrArrayRole) for _, r in result_ds.roles.items())
    assert result_ds.shape[0] == source_data.shape[0]


@pytest.mark.skip
def test_fillna_medians(spark: SparkSession):
    source_data = pd.DataFrame(data={
        "a": [0.1, 34.7, float("nan"), 2.01, 5.0],
        "b": [0.12, 1.7, 28.38, 0.002, 1.4],
        "c": [0.11, 12.67, 89.1, float("nan"), -0.99],
        "d": [0.001, 0.003, 0.5, 0.991, 0.1]
    })

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})
    _, spark_np_ds = compare_by_metadata(spark, ds, FillnaMedian(), SparkFillnaMedian())

    assert ~np.isnan(spark_np_ds.data).all()


@pytest.mark.skip
def test_logodds(spark: SparkSession):
    source_data = pd.DataFrame(data={
        "a": [0.1, 34.7, float(1e-10), 2.01, 5.0],
        "b": [0.12, 1.7, 28.38, 0.002, 1.4],
        "c": [0.11, 12.67, 89.1, 500.0, -0.99],
        "d": [0.001, 0.003, 0.5, 0.991, 0.1]
    })

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})
    compare_by_content(spark, ds, LogOdds(), SparkLogOdds())

@pytest.mark.skip
def test_standard_scaler(spark: SparkSession):
    source_data = pd.DataFrame(data={
        "a": [0.1, 34.7, 23.12, 2.01, 5.0],
        "b": [0.12, 1.7, 28.38, 0.002, 1.4],
        "c": [0.11, 12.67, 89.1, 500.0, -0.99],
        "d": [0.001, 0.003, 0.5, 0.991, 0.1]
    })

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})
    _, spark_np_ds = compare_by_metadata(spark, ds, StandardScaler(), SparkStandardScaler())

    assert ~np.isnan(spark_np_ds.data).all()


@pytest.mark.skip
def test_quantile_binning(spark: SparkSession):
    n_bins = 10
    source_data = pd.DataFrame(data={
        "a": [0.1, 34.7, 23.12, 2.01, 5.0],
        "b": [0.12, 1.7, 28.38, 0.002, float("nan")],
        "c": [0.11, 12.67, 89.1, 500.0, -0.99],
        "d": [0.001, 0.003, 0.5, 0.991, 0.1]
    })

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})
    lama_np_ds, spark_np_ds = compare_by_metadata(spark, ds, QuantileBinning(n_bins), SparkQuantileBinning(n_bins))
    # TODO: add more advanced check

    assert ~np.isnan(spark_np_ds.data).all()
    assert (spark_np_ds.data <= n_bins).all()
    assert (spark_np_ds.data >= 0).all()


def test_ohe(spark: SparkSession):
    make_sparse = False
    source_data = pd.DataFrame(data={
        "a": [1, 4, 5, 4, 2, 3],
        "b": [1, 4, 4, 4, 2, 3],
        "c": [1, 1, 1, 1, 1, 1],
        "d": [3, 1, 3, 2, 2, 1]
    })

    ds = PandasDataset(source_data, roles={name: CategoryRole(dtype=np.int32, label_encoded=True) for name in source_data.columns})
    _, _ = compare_by_metadata(spark, ds, OHEEncoder(make_sparse), SparkOHEEncoder(make_sparse))

