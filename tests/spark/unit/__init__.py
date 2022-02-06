import time
from copy import copy
from typing import Tuple, get_args, cast, List, Optional, Dict

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from lightautoml.dataset.np_pd_dataset import PandasDataset, NumpyDataset
from lightautoml.dataset.roles import ColumnRole, CategoryRole
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.transformers.base import SparkTransformer
from lightautoml.transformers.base import LAMLTransformer
from lightautoml.transformers.numeric import NumpyTransformable


# NOTE!!!
# All tests require PYSPARK_PYTHON env variable to be set
# for example: PYSPARK_PYTHON=/home/nikolay/.conda/envs/LAMA/bin/python


@pytest.fixture(scope="session")
def spark() -> SparkSession:

    spark = (
        SparkSession
        .builder
        .appName("LAMA-test-app")
        .master("local[1]")
        # .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .getOrCreate()
    )

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    # time.sleep(600)
    spark.stop()


@pytest.fixture(scope="session")
def spark_with_deps() -> SparkSession:
    spark = SparkSession.builder.appName("LAMA-test-app")\
        .master("local[1]") \
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


def compare_transformers_results(spark: SparkSession,
                                 ds: PandasDataset,
                                 t_lama: LAMLTransformer,
                                 t_spark: SparkTransformer,
                                 compare_metadata_only: bool = False) -> Tuple[NumpyDataset, NumpyDataset]:
    """
    Args:
        spark: session to be used for calculating the example
        ds: a dataset to be transformered by LAMA and Spark transformers
        t_lama: LAMA's version of the transformer
        t_spark: spark's version of the transformer
        compare_metadata_only: if True comapre only metadata of the resulting pair of datasets - columns
        count and their labels (e.g. features), roles and shapez

    Returns:
        A tuple of (LAMA transformed dataset, Spark transformed dataset)
    """
    sds = from_pandas_to_spark(ds, spark, ds.target)

    t_lama.fit(ds)
    transformed_ds = t_lama.transform(ds)

    # print(f"Transformed LAMA: {transformed_ds.data}")

    assert isinstance(transformed_ds, get_args(NumpyTransformable)), \
        f"The returned dataset doesn't belong numpy covertable types {NumpyTransformable} and " \
        f"thus cannot be checked againt the resulting spark dataset." \
        f"The dataset's type is {type(transformed_ds)}"

    lama_np_ds = cast(NumpyTransformable, transformed_ds).to_numpy()

    print(f"\nTransformed LAMA: \n{lama_np_ds}")
    # for row in lama_np_ds:
    #     print(row)

    t_spark.fit(sds)
    transformed_sds = t_spark.transform(sds)

    spark_np_ds = transformed_sds.to_numpy()
    print(f"\nTransformed SPRK: \n{spark_np_ds}")
    # for row in spark_np_ds:
    #     print(row)

    # One can compare lists, sets and dicts in Python using '==' operator
    # For dicts, for instance, pythons checks presence of the same keya in both dicts
    # and then compare values with the same keys in both dicts using __eq__ operator of the entities
    # https://hg.python.org/cpython/file/6f535c725b27/Objects/dictobject.c#l1839
    # https://docs.pytest.org/en/6.2.x/example/reportingdemo.html#tbreportdemo

    # compare independent of feature ordering
    assert list(sorted(lama_np_ds.features)) == list(sorted(spark_np_ds.features)), \
        f"List of features are not equal\n" \
        f"LAMA: {sorted(lama_np_ds.features)}\n" \
        f"SPARK: {sorted(spark_np_ds.features)}"

    # compare roles equality for the columns
    assert lama_np_ds.roles == spark_np_ds.roles, "Roles are not equal"

    # compare shapes
    assert lama_np_ds.shape == spark_np_ds.shape, "Shapes are not equals"

    if not compare_metadata_only:
        features: List[int] = [i for i, _ in sorted(enumerate(transformed_ds.features), key=lambda x: x[1])]

        trans_data: np.ndarray = lama_np_ds.data
        trans_data_result: np.ndarray = spark_np_ds.data
        # TODO: fix type checking here
        # compare content equality of numpy arrays
        assert np.allclose(trans_data[:, features], trans_data_result[:, features], equal_nan=True), \
            f"Results of the LAMA's transformer and the Spark based transformer are not equal: " \
            f"\n\nLAMA: \n{trans_data}" \
            f"\n\nSpark: \n{trans_data_result}"

    return lama_np_ds, spark_np_ds


def compare_by_content(spark: SparkSession,
                       ds: PandasDataset,
                       t_lama: LAMLTransformer,
                       t_spark: SparkTransformer) -> Tuple[NumpyDataset, NumpyDataset]:
    """
        Args:
            spark: session to be used for calculating the example
            ds: a dataset to be transformered by LAMA and Spark transformers
            t_lama: LAMA's version of the transformer
            t_spark: spark's version of the transformer

        Returns:
            A tuple of (LAMA transformed dataset, Spark transformed dataset)
        """
    return compare_transformers_results(spark, ds, t_lama, t_spark, compare_metadata_only=False)


def compare_by_metadata(spark: SparkSession,
                        ds: PandasDataset,
                        t_lama: LAMLTransformer,
                        t_spark: SparkTransformer) -> Tuple[NumpyDataset, NumpyDataset]:
    """

        Args:
            spark: session to be used for calculating the example
            ds: a dataset to be transformered by LAMA and Spark transformers
            t_lama: LAMA's version of the transformer
            t_spark: spark's version of the transformer

        Returns:
            A tuple of (LAMA transformed dataset, Spark transformed dataset)

        NOTE: Content of the datasets WON'T be checked for equality.
        This function should be used only to compare stochastic-based transformers
    """
    return compare_transformers_results(spark, ds, t_lama, t_spark, compare_metadata_only=True)


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


def from_pandas_to_spark(p: PandasDataset,
                         spark: SparkSession,
                         target: Optional[pd.Series] = None,
                         folds: Optional[pd.Series] = None,
                         task: Optional[SparkTask] = None,
                         to_vector: bool = False,
                         fill_folds_with_zeros_if_not_present: bool= False) -> SparkDataset:
    pdf = cast(pd.DataFrame, p.data)
    pdf = pdf.copy()
    pdf[SparkDataset.ID_COLUMN] = pdf.index

    roles = copy(p.roles)

    if target is not None:
        # TODO: you may have an array in the input cols, so probably it should be transformed into the vector
        tpdf = target.to_frame("target")
        tpdf[SparkDataset.ID_COLUMN] = pdf.index
    else:
        try:
            tpdf = p.target.to_frame("target")
            tpdf[SparkDataset.ID_COLUMN] = pdf.index
        except AttributeError:
            tpdf = pd.DataFrame({SparkDataset.ID_COLUMN: pdf.index, "target": np.zeros(pdf.shape[0])})

    if folds is not None:
        fpdf = folds.to_frame("folds")
        fpdf[SparkDataset.ID_COLUMN] = pdf.index
    else:
        try:
            fpdf = p.folds.to_frame("folds")
            fpdf[SparkDataset.ID_COLUMN] = pdf.index
        except AttributeError:
            fpdf = pd.DataFrame({SparkDataset.ID_COLUMN: pdf.index, "folds": np.zeros(pdf.shape[0])}) \
                if fill_folds_with_zeros_if_not_present else None

    target_sdf = spark.createDataFrame(data=tpdf)
    # target_sdf = target_sdf.fillna(0.0)

    obj_columns = list(pdf.select_dtypes(include=['object']))
    pdf[obj_columns] = pdf[obj_columns].astype(str)
    sdf = spark.createDataFrame(data=pdf)
    # sdf = sdf.fillna(0.0)

    if to_vector:
        cols = [c for c in pdf.columns if c != SparkDataset.ID_COLUMN]
        # TODO: cols[0] should be fixed
        general_feat = cols[0]
        sdf = sdf.select(SparkDataset.ID_COLUMN, F.array(*cols).alias(general_feat))
        roles = {general_feat: NumericVectorOrArrayRole(len(cols), f"{general_feat}_{{}}", dtype=roles[cols[0]].dtype)}

    kwargs = dict()
    if fpdf is not None:
        folds_sdf = spark.createDataFrame(data=fpdf)
        kwargs["folds"] = folds_sdf

    return SparkDataset(sdf, roles=roles, target=target_sdf, task=task if task else p.task, **kwargs)


def compare_obtained_datasets(lama_ds: NumpyDataset, spark_ds: SparkDataset):
    lama_np_ds = cast(NumpyTransformable, lama_ds).to_numpy()
    spark_np_ds = spark_ds.to_numpy()

    assert list(sorted(lama_np_ds.features)) == list(sorted(spark_np_ds.features)), \
        f"List of features are not equal\n" \
        f"LAMA: {sorted(lama_np_ds.features)}\n" \
        f"SPARK: {sorted(spark_np_ds.features)}"

    # compare roles equality for the columns
    assert lama_np_ds.roles == spark_np_ds.roles, \
        f"Roles are not equal.\n" \
        f"LAMA: {lama_np_ds.roles}\n" \
        f"Spark: {spark_np_ds.roles}"

    # compare shapes
    assert lama_np_ds.shape == spark_np_ds.shape

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
