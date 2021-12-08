# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightautoml.spark.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.tasks.base import Task
from lightautoml.spark.utils import spark_session

import numpy as np

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    with spark_session(parallelism=4) as spark:
        # load and prepare data
        data = pd.read_csv("../data/sampled_app_train.csv")


        train_data, test_data = train_test_split(data, test_size=0.2, stratify=data["TARGET"], random_state=42)

        # run automl
        automl = TabularAutoML(
            spark=spark,
            task=Task("binary"),
            general_params={"use_algos": ["lgb", "linear_l2"]}
        )
        oof_predictions = automl.fit_predict(train_data, roles={"target": "TARGET", "drop": ["SK_ID_CURR"]})
        # raise
        te_pred = automl.predict(test_data)

        # calculate scores
        print(f"Score for out-of-fold predictions: {roc_auc_score(train_data['TARGET'].values, oof_predictions.data[:, 0])}")
        print(f"Score for hold-out: {roc_auc_score(test_data['TARGET'].values, te_pred.data[:, 0])}")
