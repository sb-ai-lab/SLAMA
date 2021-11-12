from typing import Tuple, get_args, cast, List

import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset, NumpyDataset
from lightautoml.spark.transformers.base import SparkTransformer
from lightautoml.spark.utils import from_pandas_to_spark
from lightautoml.transformers.base import LAMLTransformer
from lightautoml.transformers.numeric import NumpyTransformable

import numpy as np


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
    sds = from_pandas_to_spark(ds, spark)

    t_lama.fit(ds)
    transformed_ds = t_lama.transform(ds)

    assert isinstance(transformed_ds, get_args(NumpyTransformable)), \
        f"The returned dataset doesn't belong numpy covertable types {NumpyTransformable} and " \
        f"thus cannot be checked againt the resulting spark dataset." \
        f"The dataset's type is {type(transformed_ds)}"

    lama_np_ds = cast(NumpyTransformable, transformed_ds).to_numpy()

    t_spark.fit(sds)
    transformed_sds = t_spark.transform(sds)

    spark_np_ds = transformed_sds.to_numpy()

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
        diff = (trans_data[:, features] - trans_data_result[:, features])
        assert (diff < 0.001).all(), \
            f"Results of the LAMA's transformer and the Spark based transformer are not equal: " \
            f"\n\nLAMA: \n{trans_data}" \
            f"\n\nSpark: \n{trans_data_result}" \
            f"\n\nDiff: \n{diff}" \
            f"\n\nCompare matrix: \n{diff < 0.001}"

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
