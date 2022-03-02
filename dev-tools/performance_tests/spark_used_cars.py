# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging.config
import os
import pickle
import shutil
from typing import Dict, Any, Optional, Tuple, cast

import yaml
from pyspark.sql import functions as F, SparkSession

from dataset_utils import datasets
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.utils import spark_session, log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT
from lightautoml.spark.validation.iterators import SparkFoldsIterator
from lightautoml.utils.tmp_utils import log_data, LAMA_LIBRARY

logger = logging.getLogger(__name__)

DUMP_METADATA_NAME = "metadata.pickle"
DUMP_DATA_NAME = "data.parquet"


def dump_data(path: str, ds: SparkDataset, **meta_kwargs):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)

    metadata_file = os.path.join(path, DUMP_METADATA_NAME)
    data_file = os.path.join(path, DUMP_DATA_NAME)

    metadata = {
        "roles": ds.roles,
        "target": ds.target_column,
        "folds": ds.folds_column,
        "task_name": ds.task.name if ds.task else None
    }
    metadata.update(meta_kwargs)

    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    ds.data.write.parquet(data_file)


def load_dump_if_exist(spark: SparkSession, path: str) -> Optional[Tuple[SparkDataset, Dict]]:
    if not os.path.exists(path):
        return None

    metadata_file = os.path.join(path, DUMP_METADATA_NAME)
    data_file = os.path.join(path, DUMP_DATA_NAME)

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    df = spark.read.parquet(data_file).repartition(16).cache()
    df.write.mode('overwrite').format('noop').save()

    ds = SparkDataset(
        data=df,
        roles=metadata["roles"],
        task=SparkTask(metadata["task_name"]),
        target=metadata["target"],
        folds=metadata["folds"]
    )

    return ds, metadata


def prepare_test_and_train(spark: SparkSession, path:str, seed: int) -> Tuple[SparkDataFrame, SparkDataFrame]:
    data = spark.read.csv(path, header=True, escape="\"")  # .repartition(4)

    data = data.select(
        '*',
        F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
        F.rand(seed).alias('is_test')
    ).cache()
    data.write.mode('overwrite').format('noop').save()
    # train_data, test_data = data.randomSplit([0.8, 0.2], seed=seed)

    train_data = data.where(F.col('is_test') < 0.8).drop('is_test').cache()
    test_data = data.where(F.col('is_test') >= 0.8).drop('is_test').cache()

    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    return train_data, test_data


def calculate_automl(path: str,
                     task_type: str,
                     metric_name: str,
                     target_col: str = 'target',
                     seed: int = 42,
                     cv: int = 5,
                     use_algos = ("lgb", "linear_l2"),
                     roles: Optional[Dict] = None,
                     dtype: Optional[None] = None,
                     spark_config: Optional[Dict[str, Any]] = None,
                     **_) -> Dict[str, Any]:
    roles = roles if roles else {}

    os.environ[LAMA_LIBRARY] = "spark"
    if not spark_config:
        spark_args = {"master": "local[4]"}
    else:
        spark_args = {'session_args': spark_config}

    with spark_session(**spark_args) as spark:
        with log_exec_timer("spark-lama training") as train_timer:
            task = SparkTask(task_type)
            train_data, test_data = prepare_test_and_train(spark, path, seed)

            test_data_dropped = test_data

            automl = SparkTabularAutoML(
                spark=spark,
                task=task,
                general_params={"use_algos": use_algos},
                reader_params={"cv": cv, "advanced_roles": False},
                tuning_params={'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
            )

            oof_predictions = automl.fit_predict(
                train_data,
                roles=roles
            )

        logger.info("Predicting on out of fold")

        score = task.get_dataset_metric()
        metric_value = score(oof_predictions)

        logger.info(f"{metric_name} score for out-of-fold predictions: {metric_value}")

        with log_exec_timer("spark-lama predicting on test") as predict_timer:
            te_pred = automl.predict(test_data_dropped, add_reader_attrs=True)

            score = task.get_dataset_metric()
            test_metric_value = score(te_pred)

            logger.info(f"{metric_name} score for test predictions: {test_metric_value}")

        logger.info("Predicting is finished")

        return {"metric_value": metric_value, "test_metric_value": test_metric_value,
                "train_duration_secs": train_timer.duration,
                "predict_duration_secs": predict_timer.duration}


def calculate_lgbadv_boostlgb(
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        spark_config: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        **_) -> Dict[str, Any]:
    roles = roles if roles else {}

    os.environ[LAMA_LIBRARY] = "spark"
    if not spark_config:
        spark_args = {"master": "local[4]"}
    else:
        spark_args = {'session_args': spark_config}

    with spark_session(**spark_args) as spark:
        with log_exec_timer("spark-lama ml_pipe") as pipe_timer:
            # chkp = load_dump_if_exist(spark, checkpoint_path) if checkpoint_path else None
            chkp = None
            if not chkp:
                task = SparkTask(task_type)
                train_data, test_data = prepare_test_and_train(spark, path, seed)

                sreader = SparkToSparkReader(task=task, cv=3, advanced_roles=False)
                sdataset = sreader.fit_read(train_data, roles=roles)

                ml_alg_kwargs = {
                    'auto_unique_co': 10,
                    'max_intersection_depth': 3,
                    'multiclass_te_co': 3,
                    'output_categories': True,
                    'top_intersections': 4
                }

                iterator = SparkFoldsIterator(sdataset, n_folds=cv)
                lgb_features = SparkLGBAdvancedPipeline(**ml_alg_kwargs)

                # # Process features and train the model
                iterator = iterator.apply_feature_pipeline(lgb_features)

                if checkpoint_path is not None:
                    ds = cast(SparkDataset, iterator.train)
                    dump_data(checkpoint_path, ds, iterator_input_roles=iterator.input_roles)
            else:
                chkp_ds, metadata = chkp
                iterator = SparkFoldsIterator(chkp_ds, n_folds=cv)
                iterator.input_roles = metadata['iterator_input_roles']

            spark_ml_algo = SparkBoostLGBM(cacher_key='main_cache', freeze_defaults=False)
            spark_ml_algo, _ = tune_and_fit_predict(spark_ml_algo, DefaultTuner(), iterator)

        return {pipe_timer.name: pipe_timer.duration}


if __name__ == "__main__":
    logging.config.dictConfig(logging_config(level=logging.INFO, log_filename="/tmp/lama.log"))
    logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
    logger = logging.getLogger(__name__)

    # Read values from config file
    with open("/scripts/config.yaml", "r") as stream:
        config_data = yaml.safe_load(stream)

    func_name = config_data['func']
    ds_cfg = datasets()[config_data['dataset']]
    del config_data['func']
    del config_data['dataset']
    ds_cfg.update(config_data)

    if func_name == "calculate_automl":
        func = calculate_automl
    elif func_name == "calculate_lgbadv_boostlgb":
        func = calculate_lgbadv_boostlgb
    else:
        raise ValueError(f"Incorrect func name: {func_name}. "
                         f"Only the following are supported: "
                         f"{['calculate_automl', 'calculate_lgbadv_boostlgb']}")

    result = func(**ds_cfg)
    print(f"EXP-RESULT: {result}")
