import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from typing import cast, List
from lightautoml.transformers.numeric import NumpyTransformable
from lightautoml.tasks import Task
from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.spark.transformers.categorical import LabelEncoder as SparkLabelEncoder, \
    OrdinalEncoder as SparkOrdinalEncoder, CatIntersectstions as SparkCatIntersectstions
from lightautoml.transformers.categorical import LabelEncoder, OrdinalEncoder, CatIntersectstions
from lightautoml.transformers.base import SequentialTransformer, UnionTransformer
from lightautoml.spark.transformers.base import SequentialTransformer as SparkSequentialTransformer, \
    UnionTransformer as SparkUnionTransformer
from . import DatasetForTest, spark, compare_obtained_datasets

from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.spark.pipelines.features.linear_pipeline import LinearFeatures as SparkLinearFeatures

from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.spark.pipelines.features.lgb_pipeline import LGBSimpleFeatures as SparkLGBSimpleFeatures

from lightautoml.spark.dataset.base import SparkDataset

DATASETS = [

    # DatasetForTest("test_transformers/resources/datasets/dataset_23_cmc.csv", default_role=CategoryRole(np.int32)),

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
def test_linear_features(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))
    sds = SparkDataset.from_lama(ds, spark)

    linear_features = LinearFeatures(
        output_categories=True,
        top_intersections=4,
        max_intersection_depth=3,
        subsample=100000,
        auto_unique_co=50,
        multiclass_te_co=3
    )

    lama_transformer = linear_features.create_pipeline(ds)
    lama_ds = lama_transformer.fit_transform(ds)

    spark_linear_features = SparkLinearFeatures(
        output_categories=True,
        top_intersections=4,
        max_intersection_depth=3,
        subsample=100000,
        auto_unique_co=50,
        multiclass_te_co=3
    )

    spark_transformer = spark_linear_features.create_pipeline(sds)
    spark_ds = spark_transformer.fit_transform(sds)


    compare_obtained_datasets(lama_ds, spark_ds)


@pytest.mark.parametrize("dataset", DATASETS)
def test_lgb_simple_features(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))
    sds = SparkDataset.from_lama(ds, spark)

    lgb_features = LGBSimpleFeatures()

    lama_transformer = lgb_features.create_pipeline(ds)
    lama_ds = lama_transformer.fit_transform(ds)

    spark_lgb_features = SparkLGBSimpleFeatures()

    spark_transformer = spark_lgb_features.create_pipeline(sds)
    spark_ds = spark_transformer.fit_transform(sds)

    compare_obtained_datasets(lama_ds, spark_ds)
