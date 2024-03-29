#!/usr/bin/env python
# coding: utf-8

import logging.config
import time

import numpy as np
import pandas as pd

from examples_utils import get_spark_session
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.selection.base import ComposedSelector
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector
from lightautoml.pipelines.selection.importance_based import (
    ModelBasedImportanceEstimator,
)
from lightautoml.pipelines.selection.permutation_importance_based import (
    NpIterativeFeatureSelector,
)
from sklearn.model_selection import train_test_split

from sparklightautoml.automl.base import SparkAutoML
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.pipelines.selection.base import BugFixSelectionPipelineWrapper
from sparklightautoml.pipelines.selection.base import SparkSelectionPipelineWrapper
from sparklightautoml.pipelines.selection.permutation_importance_based import (
    SparkNpPermutationImportanceEstimator,
)
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import VERBOSE_LOGGING_FORMAT
from sparklightautoml.utils import logging_config


logging.config.dictConfig(logging_config(level=logging.INFO, log_filename="/tmp/slama.log"))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


if __name__ == "__main__":
    spark = get_spark_session()

    np.random.seed(42)

    logger.info("Load data...")
    data = pd.read_csv("examples/data/sampled_app_train.csv")

    # for test purposes we may reduce number of active columns to use
    required_cols = ["TARGET", "DAYS_BIRTH", "DAYS_EMPLOYED"]
    cols = [c for c in data.columns if c not in required_cols][:2]
    data = data[cols + required_cols]

    logger.info("Data loaded")

    logger.info("Features modification from user side...")
    data["BIRTH_DATE"] = (np.datetime64("2018-01-01") + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))).astype(
        str
    )
    data["EMP_DATE"] = (
        np.datetime64("2018-01-01") + np.clip(data["DAYS_EMPLOYED"], None, 0).astype(np.dtype("timedelta64[D]"))
    ).astype(str)

    data["constant"] = 1
    data["allnan"] = np.nan

    data.drop(["DAYS_BIRTH", "DAYS_EMPLOYED"], axis=1, inplace=True)
    logger.info("Features modification finished")

    logger.info("Split data...")
    train_data, test_data = train_test_split(data, test_size=2000, stratify=data["TARGET"], random_state=13)
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    logger.info("Data splitted. Parts sizes: train_data = {}, test_data = {}".format(train_data.shape, test_data.shape))

    train_data_sdf = spark.createDataFrame(train_data).cache()
    test_data_sdf = spark.createDataFrame(test_data).cache()
    train_data_sdf.write.mode("overwrite").format("noop").save()
    test_data_sdf.write.mode("overwrite").format("noop").save()

    logger.info("Create task..")
    task = SparkTask("binary")
    logger.info("Task created")

    logger.info("Create reader...")
    sreader = SparkToSparkReader(task=task, cv=5, random_state=1, advanced_roles=False)
    logger.info("Reader created")

    # selector parts
    logger.info("Create feature selector")
    model01 = SparkBoostLGBM(
        default_params={
            "learningRate": 0.05,
            "numLeaves": 64,
        }
    )
    model02 = SparkBoostLGBM(
        default_params={
            "learningRate": 0.05,
            "numLeaves": 64,
        }
    )
    pipe0 = SparkLGBSimpleFeatures()
    pie = SparkNpPermutationImportanceEstimator()
    pie1 = ModelBasedImportanceEstimator()
    sel1 = ImportanceCutoffSelector(pipe0, model01, pie1, cutoff=0)
    sel2 = NpIterativeFeatureSelector(pipe0, model02, pie, feature_group_size=1, max_features_cnt_in_result=15)
    selector = BugFixSelectionPipelineWrapper(ComposedSelector([sel1, sel2]))
    logger.info("Feature selector created")

    # pipeline 1 level parts
    logger.info("Start creation pipeline_1...")
    pipe = SparkLGBSimpleFeatures()

    logger.info("\t ParamsTuner1 and Model1...")
    params_tuner1 = OptunaTuner(n_trials=1, timeout=100)
    model1 = SparkBoostLGBM(default_params={"learningRate": 0.05, "numLeaves": 128})
    logger.info("\t Tuner1 and model1 created")

    logger.info("\t ParamsTuner2 and Model2...")
    model2 = SparkBoostLGBM(
        default_params={
            "learningRate": 0.025,
            "numLeaves": 64,
        }
    )
    logger.info("\t Tuner2 and model2 created")

    logger.info("\t Pipeline1...")
    pipeline_lvl1 = SparkMLPipeline(
        ml_algos=[(model1, params_tuner1), model2],
        pre_selection=SparkSelectionPipelineWrapper(selector),
        features_pipeline=pipe,
        post_selection=None,
    )
    logger.info("Pipeline1 created")

    # pipeline 2 level parts
    logger.info("Start creation pipeline_2...")
    pipe1 = SparkLGBSimpleFeatures()

    logger.info("\t ParamsTuner and Model...")
    model = SparkBoostLGBM(
        default_params={
            "learningRate": 0.05,
            "numLeaves": 64,
            "maxBin": 1024,
            # "seed": 3,
            "numThreads": 5,
        }
    )
    logger.info("\t Tuner and model created")

    logger.info("\t Pipeline2...")
    pipeline_lvl2 = SparkMLPipeline(ml_algos=[model], pre_selection=None, features_pipeline=pipe1, post_selection=None)
    logger.info("Pipeline2 created")

    logger.info("Create AutoML pipeline...")
    automl = SparkAutoML(
        sreader,
        [
            [pipeline_lvl1],
            [pipeline_lvl2],
        ],
        skip_conn=False,
    )

    logger.info("AutoML pipeline created...")

    logger.info("Start AutoML pipeline fit_predict...")
    start_time = time.time()
    oof_pred = automl.fit_predict(train_data_sdf, roles={"target": "TARGET"}).persist()
    logger.info("AutoML pipeline fitted and predicted. Time = {:.3f} sec".format(time.time() - start_time))

    logger.info(f"Feature importances of selector:\n{selector.get_features_score()}")

    logger.info(f"oof_pred:\n{oof_pred}\nShape = {oof_pred.shape}")

    logger.info(f"Feature importances of top level algorithm:\n{automl.levels[-1][0].ml_algos[0].get_features_score()}")

    logger.info(
        f"Feature importances of lowest level algorithm "
        f"- model 0:\n{automl.levels[0][0].ml_algos[0].get_features_score()}"
    )

    logger.info(
        f"Feature importances of lowest level algorithm "
        f"- model 1:\n{automl.levels[0][0].ml_algos[1].get_features_score()}"
    )

    test_pred = automl.predict(test_data_sdf, add_reader_attrs=True)
    logger.info("Prediction for test data:\n{}\nShape = {}".format(test_pred, test_pred.shape))

    logger.info("Check scores...")
    score = task.get_dataset_metric()
    off_score = score(oof_pred)
    test_score = score(test_pred)
    logger.info(f"OOF score: {off_score}")
    logger.info(f"TEST score: {test_score}")

    oof_pred.unpersist()
    # this is necessary if persistence_manager is of CompositeManager type
    # it may not be possible to obtain oof_predictions (predictions from fit_predict) after calling unpersist_all
    automl.persistence_manager.unpersist_all()

    spark.stop()
