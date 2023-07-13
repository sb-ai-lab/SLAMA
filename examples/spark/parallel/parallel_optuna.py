import logging
from logging import config
from typing import Tuple, Union, Callable

import optuna
from lightautoml.ml_algo.tuning.optuna import TunableAlgo
from lightautoml.ml_algo.utils import tune_and_fit_predict
from pyspark.sql import functions as sf

from examples.spark.examples_utils import get_spark_session
from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.tuning.parallel_optuna import ParallelOptunaTuner
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT
from sparklightautoml.validation.base import SparkBaseTrainValidIterator
from sparklightautoml.validation.iterators import SparkFoldsIterator

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


class ProgressReportingOptunaTuner(ParallelOptunaTuner):
    def _get_objective(self,
                       ml_algo: TunableAlgo,
                       estimated_n_trials: int,
                       train_valid_iterator: SparkBaseTrainValidIterator) \
            -> Callable[[optuna.trial.Trial], Union[float, int]]:
        obj_func = super()._get_objective(ml_algo, estimated_n_trials, train_valid_iterator)

        def func(*args, **kwargs):
            obj_score = obj_func(*args, **kwargs)
            logger.info(f"Objective score: {obj_score}")
            return obj_score

        return func


def train_test_split(dataset: SparkDataset, test_slice_or_fold_num: Union[float, int] = 0.2) \
        -> Tuple[SparkDataset, SparkDataset]:

    if isinstance(test_slice_or_fold_num, float):
        assert 0 <= test_slice_or_fold_num <= 1
        train, test = dataset.data.randomSplit([1 - test_slice_or_fold_num, test_slice_or_fold_num])
    else:
        train = dataset.data.where(sf.col(dataset.folds_column) != test_slice_or_fold_num)
        test = dataset.data.where(sf.col(dataset.folds_column) == test_slice_or_fold_num)

    train_dataset, test_dataset = dataset.empty(), dataset.empty()
    train_dataset.set_data(train, dataset.features, roles=dataset.roles)
    test_dataset.set_data(test, dataset.features, roles=dataset.roles)

    return train_dataset, test_dataset


if __name__ == "__main__":
    spark = get_spark_session()

    feat_pipe = "lgb_adv"  # linear, lgb_simple or lgb_adv
    dataset_name = "lama_test_dataset"
    parallelism = 3

    # load and prepare data
    ds = SparkDataset.load(
        path=f"/tmp/{dataset_name}__{feat_pipe}__features.dataset",
        persistence_manager=PlainCachePersistenceManager()
    )
    train_ds, test_ds = train_test_split(ds, test_slice_or_fold_num=4)

    # create main entities
    computations_manager = ParallelComputationsManager(parallelism=parallelism, use_location_prefs_mode=True)
    iterator = SparkFoldsIterator(train_ds).convert_to_holdout_iterator()
    tuner = ProgressReportingOptunaTuner(
        n_trials=10,
        timeout=300,
        parallelism=parallelism,
        computations_manager=computations_manager
    )
    ml_algo = SparkBoostLGBM(default_params={"numIterations": 500}, computations_settings=computations_manager)
    score = ds.task.get_dataset_metric()

    # fit and predict
    _, oof_preds = tune_and_fit_predict(ml_algo, tuner, iterator)
    test_preds = ml_algo.predict(test_ds)

    # estimate oof and test metrics
    oof_metric_value = score(oof_preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(ds.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))

    test_metric_value = score(test_preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(ds.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))

    logger.info(f"OOF metric: {oof_metric_value}")
    logger.info(f"Test metric: {oof_metric_value}")

    spark.stop()
