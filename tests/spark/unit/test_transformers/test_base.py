import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from typing import cast, List
from lightautoml.transformers.numeric import NumpyTransformable

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.spark.transformers.categorical import LabelEncoder as SparkLabelEncoder, \
    OrdinalEncoder as SparkOrdinalEncoder, CatIntersectstions as SparkCatIntersectstions
from lightautoml.transformers.categorical import LabelEncoder, OrdinalEncoder, CatIntersectstions
from lightautoml.transformers.base import SequentialTransformer, UnionTransformer, ColumnsSelector
from lightautoml.spark.transformers.base import SequentialTransformer as SparkSequentialTransformer, \
    UnionTransformer as SparkUnionTransformer, ColumnsSelector as SparkColumnsSelector
from . import compare_by_content, from_pandas_to_spark, DatasetForTest, spark, compare_obtained_datasets
from lightautoml.spark.dataset.base import SparkDataset

DATASETS = [

    DatasetForTest("test_transformers/resources/datasets/dataset_23_cmc.csv", default_role=CategoryRole(np.int32)),

    DatasetForTest("test_transformers/resources/datasets/house_prices.csv",
                   columns=["Id", "MSSubClass", "MSZoning", "LotFrontage", "WoodDeckSF"],
                   roles={
                       "Id": CategoryRole(np.int32),
                       "MSSubClass": CategoryRole(np.int32),
                       "MSZoning": CategoryRole(str),
                       "LotFrontage": CategoryRole(np.float32),
                       "WoodDeckSF": CategoryRole(bool)
                   })
]


@pytest.mark.parametrize("dataset", DATASETS)
def test_seq_transformer(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)
    spark_ds = from_pandas_to_spark(ds, spark)

    label_encoder = LabelEncoder()
    cat_intersections = CatIntersectstions()
    ordinal_encoder = OrdinalEncoder()

    seq_transformer = SequentialTransformer([label_encoder, cat_intersections, ordinal_encoder])

    lama_output = seq_transformer.fit_transform(ds)

    spark_label_encoder = SparkLabelEncoder()
    spark_cat_intersections = SparkCatIntersectstions()
    spark_ordinal_encoder = SparkOrdinalEncoder()

    spark_seq_transformer = SparkSequentialTransformer([spark_label_encoder, spark_cat_intersections, spark_ordinal_encoder])

    spark_output = spark_seq_transformer.fit_transform(spark_ds)

    spark_np_ds = spark_output.to_numpy()

    lama_np_ds = cast(NumpyTransformable, lama_output).to_numpy()

    # compare independent of feature ordering
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
    features: List[int] = [i for i, _ in sorted(enumerate(lama_output.features), key=lambda x: x[1])]

    # compare content equality of numpy arrays
    assert np.allclose(lama_data[:, features], spark_data[:, features], equal_nan=True), \
        f"Results of the LAMA's transformer and the Spark based transformer are not equal: " \
        f"\n\nLAMA: \n{lama_data}" \
        f"\n\nSpark: \n{spark_data}"


@pytest.mark.parametrize("dataset", DATASETS)
def test_union_transformer(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)
    spark_ds = from_pandas_to_spark(ds, spark)

    label_encoder = LabelEncoder()
    cat_intersections = CatIntersectstions()
    ordinal_encoder = OrdinalEncoder()

    union_transformer = UnionTransformer([label_encoder, cat_intersections, ordinal_encoder])

    lama_output = union_transformer.fit_transform(ds)

    spark_label_encoder = SparkLabelEncoder()
    spark_cat_intersections = SparkCatIntersectstions()
    spark_ordinal_encoder = SparkOrdinalEncoder()

    spark_union_transformer = SparkUnionTransformer([spark_label_encoder, spark_cat_intersections, spark_ordinal_encoder])

    spark_output = spark_union_transformer.fit_transform(spark_ds)

    spark_np_ds = spark_output.to_numpy()

    lama_np_ds = cast(NumpyTransformable, lama_output).to_numpy()

    # compare independent of feature ordering
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
    features: List[int] = [i for i, _ in sorted(enumerate(lama_output.features), key=lambda x: x[1])]

    # compare content equality of numpy arrays
    assert np.allclose(
        np.sort(lama_data[:, features], axis=0), np.sort(spark_data[:, features], axis=0),
        equal_nan=True
    ), \
        f"Results of the LAMA's transformer and the Spark based transformer are not equal: " \
        f"\n\nLAMA: \n{lama_data}" \
        f"\n\nSpark: \n{spark_data}"


@pytest.mark.parametrize("dataset", [DATASETS[1]])
def test_column_selector(spark: SparkSession, dataset: DatasetForTest):
    ds = PandasDataset(dataset.dataset, roles=dataset.roles)
    sds = SparkDataset.from_lama(ds, spark)

    selector = ColumnsSelector(ds.features)
    selector.fit(ds)
    lama_output = selector.transform(ds)

    spark_selector = SparkColumnsSelector(sds.features)
    spark_selector.fit(sds)
    spark_output = spark_selector.transform(sds)

    assert lama_output.features == spark_output.features




