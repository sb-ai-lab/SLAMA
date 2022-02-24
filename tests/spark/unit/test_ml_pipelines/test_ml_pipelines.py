import pickle
from typing import cast

from pyspark.sql.session import SparkSession

from lightautoml.dataset.np_pd_dataset import NumpyDataset, PandasDataset
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.ml_algo.linear_pyspark import LinearLBFGS
from lightautoml.spark.pipelines.features.lgb_pipeline import LGBSimpleFeatures, LGBAdvancedPipeline
from lightautoml.spark.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.spark.pipelines.ml.nested_ml_pipe import SparkNestedTabularMLPipeline as SparkNestedTabularMLPipeline
from lightautoml.spark.tasks.base import SparkTask

import pandas as pd

from lightautoml.validation.base import DummyIterator
from .. import from_pandas_to_spark, spark_with_deps

spark = spark_with_deps


def test_nested_tabular_ml_pipeline_with_linear_bgfs(spark: SparkSession):
    with open("unit/resources/datasets/dump_tabular_automl_lgb_linear/Lvl_1_Pipe_0_before_pre_selection.pickle", "rb") as f:
        data, target, features, roles = pickle.load(f)

    pds = PandasDataset(data, roles, task=SparkTask("binary"))
    target = pd.Series(target)
    sds = from_pandas_to_spark(pds, spark, target)

    iterator = DummyIterator(train=sds)

    # dumped from simple_tabular_classification.py
    linear_feat_kwargs = {
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

    linear_l2_model = LinearLBFGS()
    linear_l2_feats = LinearFeatures(**linear_feat_kwargs)

    selection_feats = LGBSimpleFeatures()
    selection_gbm = SparkBoostLGBM()
    selection_gbm.set_prefix("Selector")

    importance = ModelBasedImportanceEstimator()

    pre_selector = ImportanceCutoffSelector(
        selection_feats,
        selection_gbm,
        importance
    )

    ml_pipe = SparkNestedTabularMLPipeline(
        ml_algos=[linear_l2_model],
        force_calc=True,
        pre_selection=pre_selector,
        features_pipeline=linear_l2_feats,
    )

    spark_ds = ml_pipe.fit_predict(iterator)
    spark_ds = cast(SparkDataset, spark_ds)

    res_ds = spark_ds.data.toPandas()
    pass


def test_nested_tabular_ml_pipeline_with_boost_lgbm(spark: SparkSession):
    with open("unit/resources/datasets/dump_tabular_automl_lgb_linear/Lvl_0_Pipe_0_before_pre_selection.pickle", "rb") as f:
        data, target, features, roles = pickle.load(f)

    pds = PandasDataset(data, roles, task=SparkTask("binary"))
    target = pd.Series(target)
    sds = from_pandas_to_spark(pds, spark, target)

    iterator = DummyIterator(train=sds)

    ml_alg_kwargs = {
        'auto_unique_co': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': False,
        # 'subsample': 100000,
        'top_intersections': 4
    }

    ml_model = SparkBoostLGBM()
    ml_model_feats = LGBAdvancedPipeline(**ml_alg_kwargs)

    selection_feats = LGBSimpleFeatures()
    selection_gbm = SparkBoostLGBM()
    selection_gbm.set_prefix("Selector")

    importance = ModelBasedImportanceEstimator()

    pre_selector = ImportanceCutoffSelector(
        selection_feats,
        selection_gbm,
        importance
    )

    ml_pipe = SparkNestedTabularMLPipeline(
        ml_algos=[ml_model],
        force_calc=True,
        pre_selection=pre_selector,
        features_pipeline=ml_model_feats,
    )

    spark_ds = ml_pipe.fit_predict(iterator)
    spark_ds = cast(SparkDataset, spark_ds)

    res_ds = spark_ds.data.toPandas()
    pass