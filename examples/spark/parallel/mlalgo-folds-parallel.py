import logging
from logging import config
from typing import Tuple, Union

import os

from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from pyspark.sql import functions as sf

from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_timer
from sparklightautoml.validation.iterators import SparkFoldsIterator
from examples.spark.examples_utils import get_spark_session

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


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

    """
    available feat_pipe: linear, lgb_simple or lgb_adv
    available ml_algo: linear_l2, lgb
    feat_pipe, ml_algo_name = "linear", "linear_l2"
    """
    feat_pipe, ml_algo_name = "lgb_adv", "lgb"
    parallelism = 1
    dataset_name = os.environ.get("DATASET", "lama_test_dataset")

    # load and prepare data
    ds = SparkDataset.load(
        path=f"/tmp/{dataset_name}__{feat_pipe}__features.dataset",
        persistence_manager=PlainCachePersistenceManager()
    )
    train_ds, test_ds = train_test_split(ds, test_slice_or_fold_num=4)
    train_ds, test_ds = train_ds.persist(), test_ds.persist()

    # create main entities
    computations_manager = ParallelComputationsManager(parallelism=parallelism)
    iterator = SparkFoldsIterator(train_ds)#.convert_to_holdout_iterator()
    if ml_algo_name == "lgb":
        ml_algo = SparkBoostLGBM(experimental_parallel_mode=True, computations_settings=computations_manager)
    else:
        ml_algo = SparkLinearLBFGS(default_params={'regParam': [1e-5]}, computations_settings=computations_manager)

    score = ds.task.get_dataset_metric()

    # fit and predict
    with log_exec_timer("Model fitting"):
        model, oof_preds = tune_and_fit_predict(ml_algo, DefaultTuner(), iterator)

    assert model is not None and oof_preds is not None

    with log_exec_timer("Model inference (oof)"):
        # estimate oof and test metrics
        oof_metric_value = score(oof_preds.data.select(
            SparkDataset.ID_COLUMN,
            sf.col(ds.target_column).alias('target'),
            sf.col(ml_algo.prediction_feature).alias('prediction')
        ))

    with log_exec_timer("Model inference (test)"):
        test_preds = ml_algo.predict(test_ds)
        test_metric_value = score(test_preds.data.select(
            SparkDataset.ID_COLUMN,
            sf.col(ds.target_column).alias('target'),
            sf.col(ml_algo.prediction_feature).alias('prediction')
        ))

    logger.info(f"OOF metric: {oof_metric_value}")
    logger.info(f"Test metric: {test_metric_value}")
