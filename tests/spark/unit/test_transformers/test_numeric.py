import numpy as np
import pandas as pd
import pytest

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import NumericRole
from lightautoml.transformers.numeric import FillInf
from lightautoml.transformers.numeric import FillnaMedian
from lightautoml.transformers.numeric import LogOdds
from lightautoml.transformers.numeric import NaNFlags
from lightautoml.transformers.numeric import QuantileBinning
from lightautoml.transformers.numeric import StandardScaler
from pyspark.sql import SparkSession

from sparklightautoml.transformers.numeric import SparkFillInfTransformer
from sparklightautoml.transformers.numeric import SparkFillnaMedianEstimator
from sparklightautoml.transformers.numeric import SparkLogOddsTransformer
from sparklightautoml.transformers.numeric import SparkNaNFlagsEstimator
from sparklightautoml.transformers.numeric import SparkQuantileBinningEstimator
from sparklightautoml.transformers.numeric import SparkStandardScalerEstimator

from .. import DatasetForTest
from .. import compare_sparkml_by_content
from .. import compare_sparkml_by_metadata
from .. import make_spark
from .. import spark as spark_sess


make_spark = make_spark
spark = spark_sess
# Note:
# -s means no stdout capturing thus allowing one to see what happens in reality

# IMPORTANT !
# The test requires env variable PYSPARK_PYTHON to be set
# for example: PYSPARK_PYTHON=/home/<user>/.conda/envs/LAMA/bin/python


DATASETS = [
    # DatasetForTest("unit/resources/datasets/dataset_23_cmc.csv", default_role=NumericRole(np.int32)),
    DatasetForTest(
        "tests/spark/unit/resources/datasets/house_prices.csv",
        columns=["Id", "MSSubClass", "LotFrontage"],
        roles={
            "Id": NumericRole(np.int32),
            "MSSubClass": NumericRole(np.int32),
            "LotFrontage": NumericRole(np.float32),
        },
    )
]


@pytest.mark.parametrize("dataset", DATASETS)
def test_fill_inf(spark: SparkSession, dataset: DatasetForTest):
    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_sparkml_by_content(
        spark, ds, FillInf(), SparkFillInfTransformer(input_cols=ds.features, input_roles=ds.roles)
    )


# @pytest.mark.parametrize("dataset", DATASETS)
# def test_fillna_median(spark: SparkSession, dataset: DatasetForTest):
#
#     ds = PandasDataset(dataset.dataset, roles=dataset.roles)
#
#     compare_by_content(spark, ds, FillnaMedian(), SparkFillnaMedian())


def test_nan_flags(spark: SparkSession):
    nan_rate = 0.2
    source_data = pd.DataFrame(
        data={
            "a": [None if i >= 5 else i for i in range(10)],
            "b": [None if i >= 7 else i for i in range(10)],
            "c": [None if i == 2 else i for i in range(10)],
            "d": list(range(10)),
        }
    )

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})

    compare_sparkml_by_content(
        spark,
        ds,
        NaNFlags(nan_rate),
        SparkNaNFlagsEstimator(input_cols=ds.features, input_roles=ds.roles, nan_rate=nan_rate),
    )


@pytest.mark.parametrize("dataset", DATASETS)
def test_fillna_medians(spark: SparkSession, dataset: DatasetForTest):
    # source_data = pd.DataFrame(data={
    #     "a": [0.1, 34.7, float("nan"), 2.01, 5.0],
    #     "b": [0.12, 1.7, 28.38, 0.002, 1.4],
    #     "c": [0.11, 12.67, 89.1, float("nan"), -0.99],
    #     "d": [0.001, 0.003, 0.5, 0.991, 0.1]
    # })
    # ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_sparkml_by_metadata(
        spark, ds, FillnaMedian(), SparkFillnaMedianEstimator(input_cols=ds.features, input_roles=ds.roles)
    )


def test_standard_scaler(spark: SparkSession):
    source_data = pd.DataFrame(
        data={
            "a": [0.1, 34.7, 23.12, 2.01, 5.0],
            "b": [0.12, 1.7, 28.38, 0.002, 1.4],
            "c": [0.11, 12.67, 89.1, 500.0, -0.99],
            "d": [0.001, 0.003, 0.5, 0.991, 0.1],
        }
    )

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})

    compare_sparkml_by_metadata(
        spark, ds, StandardScaler(), SparkStandardScalerEstimator(input_cols=ds.features, input_roles=ds.roles)
    )


# @pytest.mark.skip("Need to check implementation again")
def test_logodds(spark: SparkSession):
    source_data = pd.DataFrame(
        data={
            "a": [0.1, 34.7, float(1e-10), 2.01, 5.0],
            "b": [0.12, 1.7, 28.38, 0.002, 1.4],
            "c": [0.11, 12.67, 89.1, 500.0, -0.99],
            "d": [0.001, 0.003, 0.5, 0.991, 0.1],
        }
    )

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})
    # TODO: Change `rtol` when lama improves LogOdds() algorithm
    # Floating point errors:
    # https://github.com/sb-ai-lab/LightAutoML/blob/master/lightautoml/transformers/numeric.py#L214
    compare_sparkml_by_content(
        spark, ds, LogOdds(), SparkLogOddsTransformer(input_cols=ds.features, input_roles=ds.roles), rtol=1.0e-1
    )


def test_quantile_binning(spark: SparkSession):
    n_bins = 10
    source_data = pd.DataFrame(
        data={
            "a": [0.1, 34.7, 23.12, 2.01, 5.0],
            "b": [0.12, 1.7, 28.38, 0.002, float("nan")],
            "c": [0.11, 12.67, 89.1, 500.0, -0.99],
            "d": [0.001, 0.003, 0.5, 0.991, 0.1],
        }
    )

    ds = PandasDataset(source_data, roles={name: NumericRole(np.float32) for name in source_data.columns})
    compare_sparkml_by_metadata(
        spark,
        ds,
        QuantileBinning(n_bins),
        SparkQuantileBinningEstimator(input_cols=ds.features, input_roles=ds.roles, nbins=n_bins),
    )
