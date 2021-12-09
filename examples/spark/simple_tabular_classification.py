# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging
import sys

# set up logging to file
# logging.basicConfig(
#      filename='log_file_name.log',
#      level=logging.INFO,
#      format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
#      datefmt='%H:%M:%S'
#  )

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s %(name)s {%(filename)s:%(lineno)d} %(levelname)s:%(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S %p'
# )

formatter = logging.Formatter(
    fmt='%(asctime)s %(name)s {%(module)s:%(lineno)d} %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p'
)
# set up logging to console
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
# # set a format which is simpler for console use
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# console.setFormatter(formatter)
# add the handler to the root logger
# logging.getLogger('').addHandler(console)
# logger_lightautoml = logging.getLogger('lightautoml')
# logger_lightautoml.setLevel(logging.INFO)
# logger_lightautoml.addHandler(console)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console)


import sys

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.spark.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.utils import spark_session, print_exec_time

import numpy as np

# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('lightautoml')]
for logger in loggers:
    logger.setLevel(logging.DEBUG)
    # logger.addHandler(console)

if __name__ == "__main__":
    with spark_session(parallelism=4) as spark:
        # load and prepare data
        data = pd.read_csv("../data/sampled_app_train.csv")

        train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["TARGET"], random_state=42)

        # run automl
        automl = TabularAutoML(
            spark=spark,
            task=SparkTask("binary"),
            general_params={"use_algos": ["lgb", "linear_l2"]}
        )
        with print_exec_time():
            oof_predictions = automl.fit_predict(train_data, roles={"target": "TARGET", "drop": ["SK_ID_CURR"]})

        # TODO: SPARK-LAMA fix bug in SparkToSparkReader with nans processing to make it working on test data
        # # raise
        # te_pred = automl.predict(test_data)
        #
        # # calculate scores
        # print(f"Score for out-of-fold predictions: {roc_auc_score(train_data['TARGET'].values, oof_predictions.data[:, 0])}")
        # print(f"Score for hold-out: {roc_auc_score(test_data['TARGET'].values, te_pred.data[:, 0])}")
