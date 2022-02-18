# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging
import logging.config
import os
from typing import Dict, Any, Optional

from dataset_utils import datasets

import sklearn
import yaml
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from lightautoml.spark.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.utils import spark_session, logging_config, VERBOSE_LOGGING_FORMAT, log_exec_timer
from lightautoml.utils.tmp_utils import log_data, LAMA_LIBRARY

logger = logging.getLogger(__name__)


def calculate_automl(path: str,
                     task_type: str,
                     metric_name: str,
                     target_col: str = 'target',
                     seed: int = 42,
                     cv: int = 5,
                     use_algos = ("lgb", "linear_l2"),
                     roles: Optional[Dict] = None,
                     dtype: Optional[None] = None,
                     spark_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    roles = roles if roles else {}

    os.environ[LAMA_LIBRARY] = "spark"
    if not spark_config:
        spark_args = {"master": "local[4]"}
    else:
        spark_args = {'session_args': spark_config}

    with spark_session(**spark_args) as spark:
        with log_exec_timer("spark-lama training") as train_timer:
            task = SparkTask(task_type)
            data = spark.read.csv(path, header=True, escape="\"").repartition(4)

            data = (
                data
                # .withColumnRenamed(target_col, f"{target_col}_old")
                # .select('*', F.col(f"{target_col}_old").astype(DoubleType()).alias(target_col)).drop(f"{target_col}_old")
                .withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id())
                .cache()
            )
            train_data, test_data = data.randomSplit([0.8, 0.2], seed=seed)

            # test_data_dropped = test_data \
            #     .drop(F.col(target_col)).cache()
            test_data_dropped = test_data.cache()

            automl = TabularAutoML(
                spark=spark,
                task=task,
                general_params={"use_algos": use_algos},
                reader_params={"cv": cv},
                tuning_params={'fit_on_holdout': True, 'max_tuning_iter': 101, 'max_tuning_time': 3600}
            )

            oof_predictions = automl.fit_predict(
                train_data,
                roles=roles
            )

        log_data("spark_test_part", {"test": test_data.select(SparkDataset.ID_COLUMN, target_col).toPandas()})

        logger.info("Predicting on out of fold")

        predict_col = oof_predictions.features[0]
        oof_preds_for_eval = (
            oof_predictions.data
            .join(train_data, on=SparkDataset.ID_COLUMN)
            .select(SparkDataset.ID_COLUMN, target_col, predict_col)
        )

        # if task_type == "reg":
        #     evaluator = RegressionEvaluator(predictionCol=oof_predictions.features[0], labelCol=target_col,
        #                                     metricName=metric_name)
        # elif task_type == "binary":
        #     evaluator = BinaryClassificationEvaluator(rawPredictionCol=oof_predictions.features[0], labelCol=target_col,
        #                                               metricName=metric_name)
        # elif task_type == "multiclass":
        #     evaluator = MulticlassClassificationEvaluator(predictionCol=oof_predictions.features[0], labelCol=target_col,
        #                                                   metricName=metric_name)
        # else:
        #     raise ValueError(f"Task type {task_type} is not supported")
        #
        # metric_value = evaluator.evaluate(oof_preds_for_eval)
        # logger.info(f"{evaluator.getMetricName()} score for out-of-fold predictions: {metric_value}")

        if metric_name == "mse":
            evaluator = sklearn.metrics.mean_squared_error
        elif metric_name == "areaUnderROC":
            evaluator = sklearn.metrics.roc_auc_score
        else:
            raise ValueError(f"Metric {metric_name} is not supported")

        oof_preds_for_eval_pdf = oof_preds_for_eval.toPandas()
        metric_value = evaluator(oof_preds_for_eval_pdf[target_col].values, oof_preds_for_eval_pdf[predict_col].values)
        logger.info(f"{metric_name} score for out-of-fold predictions: {metric_value}")

        with log_exec_timer("spark-lama predicting on test") as predict_timer:
            te_pred = automl.predict(test_data_dropped)

            te_pred = (
                te_pred.data
                .join(test_data, on=SparkDataset.ID_COLUMN)
                .select(SparkDataset.ID_COLUMN, target_col, te_pred.features[0])
            )

            # test_metric_value = evaluator.evaluate(te_pred)
            # logger.info(f"{evaluator.getMetricName()} score for test predictions: {test_metric_value}")

            te_pred_pdf = te_pred.toPandas()
            test_metric_value = evaluator(te_pred_pdf[target_col].values, te_pred_pdf[predict_col].values)
            logger.info(f"{metric_name} score for test predictions: {test_metric_value}")

        logger.info("Predicting is finished")

        return {"metric_value": metric_value, "test_metric_value": test_metric_value,
                "train_duration_secs": train_timer.duration,
                "predict_duration_secs": predict_timer.duration}


if __name__ == "__main__":
    logging.config.dictConfig(logging_config(level=logging.INFO, log_filename="/tmp/lama.log"))
    logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
    logger = logging.getLogger(__name__)

    # Read values from config file
    with open("/scripts/config.yaml", "r") as stream:
        config_data = yaml.safe_load(stream)

    ds_cfg = datasets()[config_data['dataset']]
    del config_data['dataset']
    ds_cfg.update(config_data)

    result = calculate_automl(**ds_cfg)
    print(f"EXP-RESULT: {result}")
