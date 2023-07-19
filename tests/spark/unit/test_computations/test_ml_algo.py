import itertools
from typing import Optional

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

from sparklightautoml.computations.base import ComputationsManager
from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.computations.sequential import SequentialComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.validation.iterators import SparkFoldsIterator
from .. import dataset as spark_dataset, spark_for_function

spark = spark_for_function
dataset = spark_dataset


parallel_manager_configs = [
    ParallelComputationsManager(parallelism=parallelism, use_location_prefs_mode=use_location_prefs_mode)
    for parallelism, use_location_prefs_mode in itertools.product([1, 2, 5], [False, True])
]

manager_configs = [None, SequentialComputationsManager(), *parallel_manager_configs]

ml_algo_configs = [
    lambda: SparkBoostLGBM(use_barrier_execution_mode=True),
    lambda: SparkLinearLBFGS(default_params={'regParam': [1e-5]})
]

manager_mlalgo_configs = [
    (manager, mlalgo_builder())
    for manager, mlalgo_builder in itertools.product(manager_configs, ml_algo_configs)
]


@pytest.mark.parametrize("manager,ml_algo", manager_mlalgo_configs)
def test_ml_algo(spark: SparkSession,
                 dataset: SparkDataset,
                 manager: Optional[ComputationsManager],
                 ml_algo: SparkTabularMLAlgo):
    tv_iter = SparkFoldsIterator(dataset)

    if manager is not None:
        ml_algo.computations_manager = manager

    oof_preds = ml_algo.fit_predict(tv_iter)
    preds = ml_algo.predict(dataset)

    assert ml_algo.prediction_feature in oof_preds.features
    assert ml_algo.prediction_feature in oof_preds.data.columns
    assert ml_algo.prediction_feature in preds.features
    assert ml_algo.prediction_feature in preds.data.columns

    assert oof_preds.data.count() == dataset.data.count()
    assert preds.data.count() == dataset.data.count()

    score = dataset.task.get_dataset_metric()
    oof_metric = score(oof_preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(dataset.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))
    test_metric = score(preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(dataset.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))

    assert oof_metric > 0
    assert test_metric > 0
