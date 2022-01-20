# -*- encoding: utf-8 -*-

"""
Simple example for binary classification on tabular data.
"""
import logging
from typing import Dict, Any

import pandas as pd
from sklearn.linear_model.tests.test_ridge import _mean_squared_error_callable
from sklearn.model_selection import train_test_split

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.utils import log_exec_time
from lightautoml.tasks import Task


logger = logging.getLogger(__name__)


def calculate_automl(path: str, seed: int = 42, use_algos = ("lgb", "linear_l2")) -> Dict[str, Any]:
    with log_exec_time("LAMA"):
        data = pd.read_csv(path)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

        target_col = 'price'
        task = Task("reg")

        automl = TabularAutoML(task=task, timeout=3600 * 3, general_params={"use_algos": use_algos})

        oof_predictions = automl.fit_predict(
            train_data,
            roles={
                "target": target_col,
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                "numeric": ['latitude', 'longitude', 'mileage']
            }
        )

    metric_value = _mean_squared_error_callable(train_data[target_col].values, oof_predictions.data[:, 0])
    logger.info(f"mse score for out-of-fold predictions: {metric_value}")

    with log_exec_time():
        te_pred = automl.predict(test_data)

    test_metric_value = _mean_squared_error_callable(test_data[target_col].values, te_pred.data[:, 0])
    logger.info(f"mse score for test predictions: {test_metric_value}")

    logger.info("Predicting is finished")

    return {"metric_value": metric_value, "test_metric_value": test_metric_value}
