from typing import Optional

import pytest
from lightautoml.ml_algo.utils import tune_and_fit_predict
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

from sparklightautoml.computations.base import ComputationsManager
from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.computations.sequential import SequentialComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.tuning.parallel_optuna import ParallelOptunaTuner
from sparklightautoml.validation.iterators import SparkFoldsIterator
from .. import spark as spark_sess, dataset as spark_dataset

spark = spark_sess
dataset = spark_dataset


@pytest.mark.parametrize("manager", [
    # None,
    # SequentialComputationsManager(),
    # ParallelComputationsManager(parallelism=1, use_location_prefs_mode=False),
    ParallelComputationsManager(parallelism=2, use_location_prefs_mode=False),
    # ParallelComputationsManager(parallelism=5, use_location_prefs_mode=False),
    # ParallelComputationsManager(parallelism=1, use_location_prefs_mode=True),
    # ParallelComputationsManager(parallelism=2, use_location_prefs_mode=True),
    # ParallelComputationsManager(parallelism=5, use_location_prefs_mode=True)
])
def test_parallel_optuna_tuner(spark: SparkSession, dataset: SparkDataset, manager: Optional[ComputationsManager]):
    # create main entities
    iterator = SparkFoldsIterator(dataset).convert_to_holdout_iterator()
    count = iterator.get_validation_data().data.count()
    tuner = ParallelOptunaTuner(
        n_trials=10,
        timeout=60,
        parallelism=manager.parallelism if manager else 1,
        computations_manager=manager
    )
    ml_algo = SparkBoostLGBM(
        default_params={'numIterations': 25},
        use_single_dataset_mode=True,
        use_barrier_execution_mode=True
    )

    # fit and predict
    model, oof_preds = tune_and_fit_predict(ml_algo, tuner, iterator)
    test_preds = model.predict(dataset)

    assert ml_algo.prediction_feature in oof_preds.features
    assert ml_algo.prediction_feature in oof_preds.data.columns
    assert ml_algo.prediction_feature in test_preds.features
    assert ml_algo.prediction_feature in test_preds.data.columns

    assert oof_preds.data.count() == count
    assert test_preds.data.count() == dataset.data.count()

    score = dataset.task.get_dataset_metric()
    oof_metric = score(oof_preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(dataset.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))
    test_metric = score(test_preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(dataset.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))

    assert oof_metric > 0
    assert test_metric > 0
