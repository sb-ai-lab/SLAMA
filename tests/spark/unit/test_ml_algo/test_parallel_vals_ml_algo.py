from typing import cast, Callable

import numpy as np
import pytest
from lightautoml.dataset.base import RolesDict
from pyspark.sql import SparkSession

from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.pipelines.features.base import SparkFeaturesPipeline
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.validation.iterators import SparkFoldsIterator
from .. import spark as spark_sess
from ..dataset_utils import get_test_datasets

spark = spark_sess

ml_alg_kwargs = {
    'auto_unique_co': 10,
    'max_intersection_depth': 3,
    'multiclass_te_co': 3,
    'output_categories': True,
    'top_intersections': 4
}


@pytest.mark.skip(reason="Not implemented yet")
@pytest.mark.parametrize("fp,algo_builder", [
    (
        SparkLinearFeatures(**ml_alg_kwargs),
        lambda parallelism: SparkLinearLBFGS(default_params={'regParam': [1e-5]})
    ),
    (
        SparkLGBSimpleFeatures(),
        lambda parallelism: SparkBoostLGBM(parallelism=parallelism)
    )
])
def test_parallel_crossval(spark: SparkSession,
                           fp: SparkFeaturesPipeline,
                           algo_builder: Callable[[int], SparkTabularMLAlgo]):
    # checking for equal results with different parallelism degree
    cv = 5
    config = get_test_datasets(dataset="used_cars_dataset")[0]
    task_type = cast(str, config['task_type'])
    roles = cast(RolesDict, config['roles'])
    train_path = cast(str, config['train_path'])

    task = SparkTask(task_type)
    score = task.get_dataset_metric()
    persistence_manager = PlainCachePersistenceManager()

    train_df = spark.read.csv(train_path, header=True, escape="\"")

    sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)

    sdataset = sreader.fit_read(train_df, roles=roles, persistence_manager=persistence_manager)
    fp_sdataset = fp.fit_transform(sdataset)
    iterator = SparkFoldsIterator(fp_sdataset, n_folds=cv)

    oof_preds_dss = []
    for parallelism in [1, 2, 5]:
        algo = algo_builder(parallelism)
        oof_preds = algo.fit_predict(iterator)
        oof_preds_dss.append(oof_preds)

    oof_preds_scores = [score(oof_preds) for oof_preds in oof_preds_dss]

    assert np.allclose(oof_preds_scores[:-1], oof_preds_dss[1:])


@pytest.mark.skip(reason="Not implemented yet")
def test_parallel_timer_exceeded(spark: SparkSession):
    # TODO: PARALLEL - check for correct handling of the situation when timer is execeeded
    # 1. after the first fold (parallelism=1)
    # 2. after several folds (parallelism > 1)
    pass
