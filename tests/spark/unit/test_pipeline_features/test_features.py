import numpy as np
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures, LGBAdvancedPipeline
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.spark.pipelines.features.lgb_pipeline import \
    LGBSimpleFeatures as SparkLGBSimpleFeatures, LGBAdvancedPipeline as SparkLGBAdvancedPipeline
from lightautoml.spark.pipelines.features.linear_pipeline import LinearFeatures as SparkLinearFeatures
from lightautoml.spark.transformers.base import log_exec_time
from lightautoml.tasks import Task
from .. import DatasetForTest, from_pandas_to_spark, spark, compare_obtained_datasets


DATASETS = [

    # DatasetForTest("test_transformers/resources/datasets/dataset_23_cmc.csv", default_role=CategoryRole(np.int32)),

    DatasetForTest("unit/resources/datasets/house_prices.csv",
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

    # difference in folds ??
    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))
    # sds = SparkDataset.from_lama(ds, spark)
    sds = from_pandas_to_spark(ds, spark, ds.target)

    # dumped from simple_tabular_classification.py
    kwargs = {
        'auto_unique_co': 50,
        'feats_imp': None,
        'kwargs': {},
        'max_bin_count': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': True,
        'sparse_ohe': 'auto',
        'subsample': 100000,
        'top_intersections': 4
    }

    linear_features = LinearFeatures(**kwargs)

    lama_transformer = linear_features.create_pipeline(ds)

    spark_linear_features = SparkLinearFeatures(**kwargs)

    spark_transformer = spark_linear_features.create_pipeline(sds)

    print()
    print(lama_transformer.print_structure())
    print()
    print()
    print(lama_transformer.print_tr_types())

    print("===================================================")

    print()
    print(spark_transformer.print_structure())
    print()
    print()
    print(spark_transformer.print_tr_types())

    with log_exec_time():
        lama_ds = linear_features.fit_transform(ds).to_numpy()

    with log_exec_time():
        spark_ds = spark_linear_features.fit_transform(sds)

    # time.sleep(600)
    compare_obtained_datasets(lama_ds, spark_ds)


@pytest.mark.parametrize("dataset", DATASETS)
def test_lgb_simple_features(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))
    # sds = SparkDataset.from_lama(ds, spark)
    sds = from_pandas_to_spark(ds, spark, ds.target)

    # no args in simple_tabular_classification.py
    lgb_features = LGBSimpleFeatures()

    lama_transformer = lgb_features.create_pipeline(ds)

    spark_lgb_features = SparkLGBSimpleFeatures()
    spark_transformer = spark_lgb_features.create_pipeline(sds)

    with log_exec_time():
        lama_ds = lgb_features.fit_transform(ds)

    with log_exec_time():
        spark_ds = spark_lgb_features.fit_transform(sds)

    compare_obtained_datasets(lama_ds, spark_ds)


@pytest.mark.parametrize("dataset", DATASETS)
def test_lgb_advanced_features(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))
    # sds = SparkDataset.from_lama(ds, spark)
    sds = from_pandas_to_spark(ds, spark, ds.target)

    # dumped from simple_tabular_classification.py
    kwargs = {
        'ascending_by_cardinality': False,
        'auto_unique_co': 10,
        'feats_imp': None,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': False,
        'subsample': 100000,
        'top_intersections': 4
    }

    lgb_features = LGBAdvancedPipeline(**kwargs)
    lama_transformer = lgb_features.create_pipeline(ds)

    spark_lgb_features = SparkLGBAdvancedPipeline(**kwargs)
    spark_transformer = spark_lgb_features.create_pipeline(sds)

    with log_exec_time():
        lama_ds = lgb_features.fit_transform(ds)

    with log_exec_time():
        spark_ds = spark_lgb_features.fit_transform(sds)

    compare_obtained_datasets(lama_ds, spark_ds)
