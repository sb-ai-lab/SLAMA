# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging.config
import os
import pickle
import random
import shutil
import time
from contextlib import contextmanager
from copy import copy
from typing import Dict, Any, Optional, Tuple, cast

import yaml
from pyspark import SparkFiles
from pyspark.sql import functions as F, SparkSession
from pyspark.sql.pandas.functions import pandas_udf

from dataset_utils import datasets
from lightautoml.dataset.roles import CategoryRole
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.ml_algo.base import SparkTabularMLAlgo
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.ml_algo.linear_pyspark import SparkLinearLBFGS
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline
from lightautoml.spark.pipelines.features.linear_pipeline import SparkLinearFeatures
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.transformers.categorical import SparkLabelEncoderEstimator, SparkTargetEncoderEstimator, \
    SparkCatIntersectionsEstimator
from lightautoml.spark.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT
from lightautoml.spark.validation.iterators import SparkFoldsIterator, SparkDummyIterator

import pandas as pd

logger = logging.getLogger()

DUMP_METADATA_NAME = "metadata.pickle"
DUMP_DATA_NAME = "data.parquet"


@contextmanager
def open_spark_session() -> Tuple[SparkSession, str]:
    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        spark_sess = SparkSession.builder.getOrCreate()
        config_path = SparkFiles.get('config.yaml')
    else:
        spark_sess = (
            SparkSession
            .builder
            .master("local[4]")
            .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.jar")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.driver.memory", "12g")
            .config("spark.executor.memory", "12g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            # .config("spark.eventLog.enabled", "true")
            # .config("spark.eventLog.dir", "file:///tmp/spark_logs")
            .getOrCreate()
        )
        config_path = '/tmp/config.yaml'

    spark_sess.sparkContext.setLogLevel("WARN")
    spark_sess.sparkContext.setCheckpointDir("/tmp/chkp")

    try:
        yield spark_sess, config_path
    finally:
        # wait_secs = 600
        # logger.info(f"Sleeping {wait_secs} secs before stopping")
        # time.sleep(wait_secs)
        spark_sess.stop()
        logger.info("Stopped spark session")


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

    sdf = ds.data
    cols = [F.col(c).alias(c.replace('(', '[').replace(')', ']')) for c in sdf.columns]
    sdf = sdf.select(*cols)
    sdf.write.mode('overwrite').parquet(data_file)


def load_dump_if_exist(spark: SparkSession, path: str) -> Optional[Tuple[SparkDataset, Dict]]:
    if not os.path.exists(path):
        return None

    metadata_file = os.path.join(path, DUMP_METADATA_NAME)
    data_file = os.path.join(path, DUMP_DATA_NAME)

    ex_instances = int(spark.conf.get('spark.executor.instances'))
    ex_cores = int(spark.conf.get('spark.executor.cores'))

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    df = spark.read.parquet(data_file)
    cols = [F.col(c).alias(c.replace('[', '(').replace(']', ')')) for c in df.columns]
    df = df.select(*cols).repartition(ex_instances * ex_cores).cache()

    df.write.mode('overwrite').format('noop').save()

    ds = SparkDataset(
        data=df,
        roles=metadata["roles"],
        task=SparkTask(metadata["task_name"]),
        target=metadata["target"],
        folds=metadata["folds"]
    )

    return ds, metadata


def prepare_test_and_train(spark: SparkSession, path:str, seed: int, test_proportion: float = 0.2) -> Tuple[SparkDataFrame, SparkDataFrame]:
    assert 0.0 <= test_proportion <= 1.0

    train_proportion = 1.0 - test_proportion

    data = spark.read.csv(path, header=True, escape="\"")

    # ex_instances = int(spark.conf.get('spark.executor.instances'))
    # ex_cores = int(spark.conf.get('spark.executor.cores'))

    data = data.select(
        '*',
        F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
        F.rand(seed).alias('is_test')
    ).cache()
    # ).repartition(ex_instances * ex_cores).cache()
    data.write.mode('overwrite').format('noop').save()
    # train_data, test_data = data.randomSplit([0.8, 0.2], seed=seed)

    train_data = data.where(F.col('is_test') < train_proportion).drop('is_test').cache()
    test_data = data.where(F.col('is_test') >= train_proportion).drop('is_test').cache()

    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    return train_data, test_data


def calculate_automl(
        spark: SparkSession,
        path: str,
        task_type: str,
        metric_name: str,
        seed: int = 42,
        cv: int = 5,
        use_algos = ("lgb", "linear_l2"),
        roles: Optional[Dict] = None,
        **_) -> Dict[str, Any]:
    roles = roles if roles else {}

    with log_exec_timer("spark-lama training") as train_timer:
        task = SparkTask(task_type)
        train_data, test_data = prepare_test_and_train(spark, path, seed)
        test_data_dropped = test_data

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            general_params={"use_algos": use_algos},
            lgb_params={'use_single_dataset_mode': True, "default_params": {"numIterations": 500}, "freeze_defaults": True},
            linear_l2_params={"default_params": {"regParam": [1]}},
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
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        **_) -> Dict[str, Any]:
    roles = roles if roles else {}

    # checkpoint_path = None

    with log_exec_timer("spark-lama ml_pipe") as pipe_timer:
        if checkpoint_path is not None:
            train_checkpoint_path = os.path.join(checkpoint_path, 'train.dump')
            test_checkpoint_path = os.path.join(checkpoint_path, 'test.dump')
            train_chkp = load_dump_if_exist(spark, train_checkpoint_path)
            test_chkp = load_dump_if_exist(spark, test_checkpoint_path)
        else:
            train_checkpoint_path = None
            test_checkpoint_path = None
            train_chkp = None
            test_chkp = None

        task = SparkTask(task_type)

        # train_chkp = None

        if not train_chkp or not test_chkp:
            logger.info(f"Checkpoint doesn't exist on path {checkpoint_path}. Will create it.")

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

            lgb_features = SparkLGBAdvancedPipeline(**ml_alg_kwargs)
            lgb_features.input_roles = sdataset.roles
            sdataset = lgb_features.fit_transform(sdataset)

            iterator = SparkFoldsIterator(sdataset, n_folds=cv)
            iterator.input_roles = lgb_features.output_roles

            stest = sreader.read(test_data, add_array_attrs=True)
            stest = cast(SparkDataset, lgb_features.transform(stest))

            if checkpoint_path is not None:
                dump_data(train_checkpoint_path, iterator.train, iterator_input_roles=iterator.input_roles)
                dump_data(test_checkpoint_path, stest, iterator_input_roles=iterator.input_roles)
        else:
            logger.info(f"Checkpoint exists on path {checkpoint_path}. Will use it ")

            train_chkp_ds, metadata = train_chkp
            iterator = SparkFoldsIterator(train_chkp_ds, n_folds=cv)
            iterator.input_roles = metadata['iterator_input_roles']

            stest, _ = test_chkp

        iterator = iterator.convert_to_holdout_iterator()
        # iterator = SparkDummyIterator(iterator.train, iterator.input_roles)

        score = task.get_dataset_metric()

        spark_ml_algo = SparkBoostLGBM(cacher_key='main_cache', use_single_dataset_mode=True, max_validation_size=10_000)
        spark_ml_algo, oof_preds = tune_and_fit_predict(spark_ml_algo, DefaultTuner(), iterator)

        assert spark_ml_algo is not None
        assert oof_preds is not None

        spark_ml_algo = cast(SparkTabularMLAlgo, spark_ml_algo)
        oof_preds = cast(SparkDataset, oof_preds)
        oof_preds_sdf = oof_preds.data.select(
            SparkDataset.ID_COLUMN,
            F.col(oof_preds.target_column).alias('target'),
            F.col(spark_ml_algo.prediction_feature).alias("prediction")
        )
        oof_score = score(oof_preds_sdf)

        test_preds = spark_ml_algo.predict(stest)
        test_preds_sdf = test_preds.data.select(
            SparkDataset.ID_COLUMN,
            F.col(test_preds.target_column).alias('target'),
            F.col(spark_ml_algo.prediction_feature).alias("prediction")
        )
        test_score = score(test_preds_sdf)

    return {pipe_timer.name: pipe_timer.duration, 'oof_score': oof_score, 'test_score': test_score}


def calculate_linear_l2(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        **_) -> Dict[str, Any]:
    roles = roles if roles else {}

    checkpoint_path = None

    with log_exec_timer("spark-lama ml_pipe") as pipe_timer:
        if checkpoint_path is not None:
            train_checkpoint_path = os.path.join(checkpoint_path, 'train.dump')
            test_checkpoint_path = os.path.join(checkpoint_path, 'test.dump')
            train_chkp = load_dump_if_exist(spark, train_checkpoint_path)
            test_chkp = load_dump_if_exist(spark, test_checkpoint_path)
        else:
            train_checkpoint_path = None
            test_checkpoint_path = None
            train_chkp = None
            test_chkp = None

        task = SparkTask(task_type)

        if not train_chkp or not test_chkp:
            logger.info(f"Checkpoint doesn't exist on path {checkpoint_path}. Will create it.")

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

            features = SparkLinearFeatures(**ml_alg_kwargs)
            features.input_roles = sdataset.roles
            sdataset = features.fit_transform(sdataset)

            iterator = SparkFoldsIterator(sdataset, n_folds=cv)
            iterator.input_roles = features.output_roles

            stest = sreader.read(test_data, add_array_attrs=True)
            stest = cast(SparkDataset, features.transform(stest))

            if checkpoint_path is not None:
                dump_data(train_checkpoint_path, iterator.train, iterator_input_roles=iterator.input_roles)
                dump_data(test_checkpoint_path, stest, iterator_input_roles=iterator.input_roles)
        else:
            logger.info(f"Checkpoint exists on path {checkpoint_path}. Will use it ")

            train_chkp_ds, metadata = train_chkp
            iterator = SparkFoldsIterator(train_chkp_ds, n_folds=cv)
            iterator.input_roles = metadata['iterator_input_roles']

            stest, _ = test_chkp

        iterator = iterator.convert_to_holdout_iterator()
        # iterator = SparkDummyIterator(iterator.train, iterator.input_roles)

        score = task.get_dataset_metric()

        spark_ml_algo = SparkLinearLBFGS(cacher_key='main_cache')
        spark_ml_algo, oof_preds = tune_and_fit_predict(spark_ml_algo, DefaultTuner(), iterator)

        assert spark_ml_algo is not None
        assert oof_preds is not None

        spark_ml_algo = cast(SparkTabularMLAlgo, spark_ml_algo)
        oof_preds = cast(SparkDataset, oof_preds)
        oof_preds_sdf = oof_preds.data.select(
            SparkDataset.ID_COLUMN,
            F.col(oof_preds.target_column).alias('target'),
            F.col(spark_ml_algo.prediction_feature).alias("prediction")
        )
        oof_score = score(oof_preds_sdf)

        test_preds = spark_ml_algo.predict(stest)
        test_preds_sdf = test_preds.data.select(
            SparkDataset.ID_COLUMN,
            F.col(test_preds.target_column).alias('target'),
            F.col(spark_ml_algo.prediction_feature).alias("prediction")
        )
        test_score = score(test_preds_sdf)

    return {pipe_timer.name: pipe_timer.duration, 'oof_score': oof_score, 'test_score': test_score}


def calculate_reader(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        **_):

    data, _ = prepare_test_and_train(spark, path, seed, test_proportion=0.0)

    task = SparkTask(task_type)

    with log_exec_timer("Reader") as reader_timer:
        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        sdataset = sreader.fit_read(data, roles=roles)
        sdataset.data.write.mode('overwrite').format('noop').save()

    return {"reader_time": reader_timer.duration}


def calculate_le(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        **_):

    data, _ = prepare_test_and_train(spark, path, seed, test_proportion=0.0)

    task = SparkTask(task_type)

    with log_exec_timer("Reader") as reader_timer:
        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        sdataset = sreader.fit_read(data, roles=roles)

    cat_roles = {feat: role for feat, role in sdataset.roles.items() if isinstance(role, CategoryRole)}

    with log_exec_timer("SparkLabelEncoder") as le_timer:
        estimator = SparkLabelEncoderEstimator(
            input_cols=list(cat_roles.keys()),
            input_roles=cat_roles
        )

        transformer = estimator.fit(data)

    with log_exec_timer("SparkLabelEncoder transform") as le_transform_timer:
        df = transformer.transform(data)
        df.write.mode('overwrite').format('noop').save()

    return {
        "reader_time": reader_timer.duration,
        "le_fit_time": le_timer.duration,
        "le_transform_time": le_transform_timer.duration
    }


def calculate_te(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        **_):

    checkpoint_path = None

    if checkpoint_path is not None:
        checkpoint_path = os.path.join(checkpoint_path, 'data.dump')
        chkp = load_dump_if_exist(spark, checkpoint_path)
    else:
        checkpoint_path = None
        chkp = None

    task = SparkTask(task_type)

    if not chkp:
        data, _ = prepare_test_and_train(spark, path, seed, test_proportion=0.0)

        task = SparkTask(task_type)

        with log_exec_timer("Reader") as reader_timer:
            sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
            sdataset = sreader.fit_read(data, roles=roles)

        cat_roles = {feat: role for feat, role in sdataset.roles.items() if isinstance(role, CategoryRole)}

        with log_exec_timer("SparkLabelEncoder") as le_timer:
            estimator = SparkLabelEncoderEstimator(
                input_cols=list(cat_roles.keys()),
                input_roles=cat_roles
            )

            transformer = estimator.fit(sdataset.data)

        with log_exec_timer("SparkLabelEncoder transform") as le_transform_timer:
            df = transformer.transform(sdataset.data).cache()
            df.write.mode('overwrite').format('noop').save()

        df = df.select(
            SparkDataset.ID_COLUMN,
            sdataset.folds_column,
            sdataset.target_column,
            *estimator.getOutputCols()
        )
        le_ds = sdataset.empty()
        le_ds.set_data(df, estimator.getOutputCols(), estimator.getOutputRoles())

        if checkpoint_path is not None:
            dump_data(checkpoint_path, le_ds)
    else:
        logger.info(f"Checkpoint exists on path {checkpoint_path}. Will use it ")
        le_ds, _ = chkp

    with log_exec_timer("TargetEncoder") as te_timer:
        te_estimator = SparkTargetEncoderEstimator(
            input_cols=le_ds.features,
            input_roles=le_ds.roles,
            task_name=task.name,
            folds_column=le_ds.folds_column,
            target_column=le_ds.target_column
        )

        te_transformer = te_estimator.fit(le_ds.data)

    with log_exec_timer("TargetEncoder transform") as te_transform_timer:
        df = te_transformer.transform(le_ds.data)
        df.write.mode('overwrite').format('noop').save()

    if not chkp:
        res = {
            "reader_time": reader_timer.duration,
            "le_fit_time": le_timer.duration,
            "le_transform_time": le_transform_timer.duration,
            "te_fit_time": te_timer.duration,
            "te_transform_time": te_transform_timer.duration
        }
    else:
        res = {
            "te_fit_time": te_timer.duration,
            "te_transform_time": te_transform_timer.duration
        }

    return res


def calculate_cat_te(
        spark: SparkSession,
        path: str,
        task_type: str,
        seed: int = 42,
        cv: int = 5,
        roles: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        **_):

    if checkpoint_path is not None:
        checkpoint_path = os.path.join(checkpoint_path, 'data.dump')
        chkp = load_dump_if_exist(spark, checkpoint_path)
    else:
        checkpoint_path = None
        chkp = None

    task = SparkTask(task_type)

    if not chkp:
        data, _ = prepare_test_and_train(spark, path, seed, test_proportion=0.0)

        task = SparkTask(task_type)

        with log_exec_timer("Reader") as reader_timer:
            sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
            sdataset = sreader.fit_read(data, roles=roles)

        cat_roles = {feat: role for feat, role in sdataset.roles.items() if feat in ['vin', 'city', 'power', 'torque']}

        with log_exec_timer("SparkLabelEncoder") as ci_timer:
            estimator = SparkCatIntersectionsEstimator(
                input_cols=list(cat_roles.keys()),
                input_roles=cat_roles,
                max_depth=3
            )

            transformer = estimator.fit(sdataset.data)

        with log_exec_timer("SparkLabelEncoder transform") as ci_transform_timer:
            df = transformer.transform(sdataset.data).cache()
            # df.write.mode('overwrite').format('noop').save()
            df = cast(SparkDataFrame, df)
            df = df.localCheckpoint(eager=True)

        df = df.select(
            SparkDataset.ID_COLUMN,
            sdataset.folds_column,
            sdataset.target_column,
            *estimator.getOutputCols()
        )
        le_ds = sdataset.empty()
        le_ds.set_data(df, estimator.getOutputCols(), estimator.getOutputRoles())

        if checkpoint_path is not None:
            dump_data(checkpoint_path, le_ds)
    else:
        logger.info(f"Checkpoint exists on path {checkpoint_path}. Will use it ")
        le_ds, _ = chkp

    with log_exec_timer("TargetEncoder") as te_timer:
        te_estimator = SparkTargetEncoderEstimator(
            input_cols=le_ds.features,
            input_roles=le_ds.roles,
            task_name=task.name,
            folds_column=le_ds.folds_column,
            target_column=le_ds.target_column
        )

        te_transformer = te_estimator.fit(le_ds.data)

    with log_exec_timer("TargetEncoder transform") as te_transform_timer:
        df = te_transformer.transform(le_ds.data)
        df.write.mode('overwrite').format('noop').save()

    if not chkp:
        res = {
            "reader_time": reader_timer.duration,
            "cat_fit_time": ci_timer.duration,
            "cat_transform_time": ci_transform_timer.duration,
            "te_fit_time": te_timer.duration,
            "te_transform_time": te_transform_timer.duration
        }
    else:
        res = {
            "te_fit_time": te_timer.duration,
            "te_transform_time": te_transform_timer.duration
        }

    return res


def empty_calculate(spark: SparkSession, **_):
    logger.info("Success")
    return {"result": "success"}


def calculate_broadcast(spark: SparkSession, **_):
    spark.sparkContext.setCheckpointDir("/tmp/chkp")

    data = [
        {"a": i, "b": i * 10, "c": i * 100}
        for i in range(100)
    ]

    df = spark.createDataFrame(data)
    df = df.cache()
    df.write.mode('overwrite').format('noop').save()

    with log_exec_timer("a") as gen_arr:
        mapping_size = 10_000_000
        bdata = {i: random.randint(0, 1000) for i in range(mapping_size)}

    print(f"Gen arr time: {gen_arr.duration}")

    with log_exec_timer("b") as bcast_timer:
        bval = spark.sparkContext.broadcast(bdata)

    print(f"Bcast time: {bcast_timer.duration}")

    @pandas_udf('int')
    def func1(col: pd.Series) -> pd.Series:
        mapping = bval.value
        # mapping = bdata

        return col.apply(lambda x: x + mapping[x] if x in mapping else 0.0)
        # return col.apply(lambda x: x + 10.0)

    df_1 = df.select([func1(c).alias(c) for c in df.columns])
    df_1 = df_1.cache()
    df_1.write.mode('overwrite').format('noop').save()

    df_1 = df_1.localCheckpoint(eager=True)

    bval.destroy()

    # df_1 = spark.createDataFrame(df_1.rdd, schema=df_1.schema, verifySchema=False)

    # df_1 = spark.createDataFrame(df_1.rdd)

    # with log_exec_timer("b") as chkp_timer:
    #     df_1 = df_1.checkpoint(eager=True)
    #
    # print(f"checkpoint time: {chkp_timer.duration}")

    @pandas_udf('int')
    def func2(col: pd.Series) -> pd.Series:
        return col.apply(lambda x: x - 10)

    df_2 = df_1#df_1.select([func2(c).alias(c) for c in df_1.columns])
    # df_2 = df_2.cache()
    df_2.write.mode('overwrite').format('noop').save()

    print("Finished")
    # time.sleep(600)


if __name__ == "__main__":
    logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename="/tmp/lama.log"))
    logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)

    with open_spark_session() as (spark, config_path):
        # Read values from config file
        with open(config_path, "r") as stream:
            config_data = yaml.safe_load(stream)

        func_name = config_data['func']

        if 'dataset' in config_data:
            ds_cfg = datasets()[config_data['dataset']]
        else:
            ds_cfg = dict()

        ds_cfg.update(config_data)

        if func_name == "calculate_automl":
            func = calculate_automl
        elif func_name == "calculate_lgbadv_boostlgb":
            func = calculate_lgbadv_boostlgb
        elif func_name == "calculate_linear_l2":
            func = calculate_linear_l2
        elif func_name == 'empty_calculate':
            func = empty_calculate
        elif func_name == 'calculate_reader':
            func = calculate_reader
        elif func_name == 'calculate_le':
            func = calculate_le
        elif func_name == 'calculate_te':
            func = calculate_te
        elif func_name == 'calculate_cat_te':
            func = calculate_cat_te
        elif func_name == 'calculate_broadcast':
            func = calculate_broadcast
        else:
            raise ValueError(f"Incorrect func name: {func_name}. "
                             f"Only the following are supported: "
                             f"{['calculate_automl', 'calculate_lgbadv_boostlgb']}")

        result = func(spark=spark, **ds_cfg)
        print(f"EXP-RESULT: {result}")
