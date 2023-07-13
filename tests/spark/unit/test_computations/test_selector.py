import pytest
from pyspark.sql import SparkSession

from sparklightautoml.computations.base import ComputationsManager
from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.computations.sequential import SequentialComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.pipelines.selection.permutation_importance_based import SparkNpPermutationImportanceEstimator
from sparklightautoml.validation.iterators import SparkFoldsIterator
from .. import spark as spark_sess, dataset as spark_dataset

spark = spark_sess
dataset = spark_dataset


@pytest.mark.parametrize("manager", [
    None,
    SequentialComputationsManager(),
    ParallelComputationsManager(parallelism=5, use_location_prefs_mode=False),
    ParallelComputationsManager(parallelism=5, use_location_prefs_mode=True)
])
def test_selector(spark: SparkSession, dataset: SparkDataset, manager: ComputationsManager):
    iterator = SparkFoldsIterator(dataset)

    ml_algo = SparkBoostLGBM(
        default_params={'numIterations': 25},
        use_barrier_execution_mode=True
    )
    ml_algo.fit(iterator.convert_to_holdout_iterator())
    preds = ml_algo.predict(dataset)

    perm_est = SparkNpPermutationImportanceEstimator(computations_settings=manager)
    perm_est.fit(iterator, ml_algo=ml_algo, preds=preds)

    feats_score = perm_est.get_features_score()
    print(f"Feature score: {feats_score}")
