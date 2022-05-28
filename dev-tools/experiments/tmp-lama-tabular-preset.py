import logging.config
import os
import uuid

import pandas as pd
from sklearn.model_selection import train_test_split

from examples_utils import get_dataset_attrs
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.spark.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT
from lightautoml.tasks import Task

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def main(dataset_name: str, seed: int):
    cv = 3

    # Algos and layers to be used during automl:
    # For example:
    # 1. use_algos = [["lgb"]]
    # 2. use_algos = [["linear_l2"]]
    # 3. use_algos = [["lgb", "linear_l2"], ["lgb"]]
    use_algos = [["lgb", "linear_l2"], ["lgb"]]

    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    with log_exec_timer("LAMA") as train_timer:
        data = pd.read_csv(path, dtype=dtype)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

        task = Task(task_type)

        num_threads = int(os.environ.get("EXEC_CORES", "4"))
        mem = int(os.environ.get("EXEC_MEM", "16"))

        automl = TabularAutoML(
            task=task,
            cpu_limit=num_threads,
            timeout=3600 * 12,
            memory_limit=mem,
            general_params={"use_algos": use_algos},
            reader_params={"cv": cv, "advanced_roles": False},
            lgb_params={
                "default_params": {"num_threads": num_threads, "numIterations": 500, "earlyStoppingRound": 5000},
                "freeze_defaults": True
            },
            linear_l2_params={"default_params": {"cs": [1e-5, 5e-5]}},
            gbm_pipeline_params={'max_intersection_depth': 2, 'top_intersections': 2},
            linear_pipeline_params={'max_intersection_depth': 2, 'top_intersections': 2},
        )

        oof_predictions = automl.fit_predict(
            train_data,
            roles=roles
        )

    logger.info("Predicting on out of fold")

    score = task.get_dataset_metric()
    metric_value = score(oof_predictions)

    logger.info(f"Score for out-of-fold predictions: {metric_value}")

    with log_exec_timer() as predict_timer:
        te_pred = automl.predict(test_data)
        te_pred.target = test_data[roles['target']]

        score = task.get_dataset_metric()
        test_metric_value = score(te_pred)

    logger.info(f"Score for test predictions: {test_metric_value}")

    logger.info("Predicting is finished")

    result = {
        "seed": seed,
        "dataset": dataset_name,
        "used_algo": str(use_algos),
        "metric_value": metric_value,
        "test_metric_value": test_metric_value,
        "train_duration_secs": train_timer.duration,
        "predict_duration_secs": predict_timer.duration
    }

    print(f"EXP-RESULT: {result}")

    return result


if __name__ == "__main__":
    # One can run:
    # 1. main(dataset_name="used_cars_dataset", seed=42)
    # 2. multirun(dataset_name="used_cars_dataset")
    ds_name = os.environ.get("DS_NAME", "used_cars_dataset")
    logger.info(f"Running with dataset {ds_name}")
    main(dataset_name=ds_name, seed=42)
