# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""

import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# load and prepare data
# TODO: put a correct path for used_cars dataset
from lightautoml.spark.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.tasks.base import Task as SparkTask
from lightautoml.spark.utils import spark_session, print_exec_time

data = pd.read_csv("../data/tiny_used_cars_data.csv")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# run automl
if __name__ == "__main__":
    with spark_session(parallelism=4) as spark:
        task = SparkTask("reg")

        automl = TabularAutoML(
            spark=spark,
            task=task,
            general_params={"use_algos": ["lgb", "linear_l2"]}
            # general_params={"use_algos": ["linear_l2"]}
            # general_params={"use_algos": ["lgb"]}
        )

        with print_exec_time():
            oof_predictions = automl.fit_predict(
                train_data,
                roles={"target": "price", "drop": ["dealer_zip", "description", "listed_date"]}
            )

        # TODO: SPARK-LAMA fix bug in SparkToSparkReader with nans processing to make it working on test data
        # te_pred = automl.predict(test_data)

        # # calculate scores
        # # TODO: replace with mse
        # #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
        # print(f"Score for out-of-fold predictions: {roc_auc_score(train_data['TARGET'].values, oof_predictions.data[:, 0])}")
        # print(f"Score for hold-out: {roc_auc_score(test_data['TARGET'].values, te_pred.data[:, 0])}")
