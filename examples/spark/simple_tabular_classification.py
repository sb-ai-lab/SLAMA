# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging.config

import pandas as pd
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split

from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.utils import spark_session, log_exec_time, logging_config, VERBOSE_LOGGING_FORMAT

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    with spark_session(master="local[4]") as spark:
        # load and prepare data
        data = pd.read_csv("../data/sampled_app_train.csv")

        train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["TARGET"], random_state=42)

        test_data = test_data.withColumn(SparkDataset.ID_COLUMN, F.monotonically_increasing_id())

        # run automl
        automl = SparkTabularAutoML(
            spark=spark,
            task=SparkTask("binary"),
            general_params={"use_algos": ["lgb", "linear_l2"]})

        with log_exec_time():
            oof_predictions = automl.fit_predict(train_data, roles={"target": "TARGET", "drop": ["SK_ID_CURR"]})

        logger.info("Predicting on out of fold")

        # TODO: SPARK-LAMA fix bug in SparkToSparkReader.read method
        # with log_exec_time():
        #     te_pred = automl.predict(test_data)
        #     test_data_pd = test_data.toPandas().data
        #     logger.info(f"Score for out-of-fold predictions:
        #     {roc_auc_score(train_data['TARGET'].values, oof_predictions.data[:, 0])}")
        #     logger.info(f"Score for hold-out: {roc_auc_score(test_data['TARGET'].values, te_pred.data[:, 0])}")
