import logging
import logging.config
import os
import shutil
from copy import deepcopy, copy
from pprint import pprint
from typing import Any, Callable, Dict

from pyspark.ml import Pipeline

from lama_used_cars import calculate_automl as lama_automl
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT
from lightautoml.utils.tmp_utils import LOG_DATA_DIR, log_config
from spark_used_cars import calculate_automl as spark_automl


def datasets() -> Dict[str, Any]:
    all_datastes = {
        "used_cars_dataset": {
            "path": "examples/data/small_used_cars_data.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "price",
            "roles": {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            },
            "dtype": {
                'fleet': 'str', 'frame_damaged': 'str',
                'has_accidents': 'str', 'isCab': 'str',
                'is_cpo': 'str', 'is_new': 'str',
                'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
            }
        },

        "lama_test_dataset": {
            "path": "./examples/data/sampled_app_train.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "TARGET",
            "roles": {"target": "TARGET", "drop": ["SK_ID_CURR"]},
        },

        # https://www.openml.org/d/734
        "ailerons_dataset": {
            "path": "/opt/ailerons.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "binaryClass",
            "roles": {"target": "binaryClass"},
        },

        # https://www.openml.org/d/4534
        "phishing_websites_dataset": {
            "path": "/opt/PhishingWebsites.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "Result",
            "roles": {"target": "Result"},
        },

        # https://www.openml.org/d/981
        "kdd_internet_usage": {
            "path": "/opt/kdd_internet_usage.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "Who_Pays_for_Access_Work",
            "roles": {"target": "Who_Pays_for_Access_Work"},
        },

        # https://www.openml.org/d/42821
        "nasa_dataset": {
            "path": "/opt/nasa_phm2008.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "class",
            "roles": {"target": "class"},
        },

        # https://www.openml.org/d/4549
        "buzz_dataset": {
            "path": "/opt/Buzzinsocialmedia_Twitter_25k.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "Annotation",
            "roles": {"target": "Annotation"},
        },

        # https://www.openml.org/d/372
        "internet_usage": {
            "path": "/opt/internet_usage.csv",
            "task_type": "multiclass",
            "metric_name": "ova",
            "target_col": "Actual_Time",
            "roles": {"target": "Actual_Time"},
        },

        # https://www.openml.org/d/4538
        "gesture_segmentation": {
            "path": "/opt/gesture_segmentation.csv",
            "task_type": "multiclass",
            "metric_name": "ova",
            "target_col": "Phase",
            "roles": {"target": "Phase"},
        },

        # https://www.openml.org/d/382
        "ipums_97": {
            "path": "/opt/ipums_97.csv",
            "task_type": "multiclass",
            "metric_name": "ova",
            "target_col": "movedin",
            "roles": {"target": "movedin"},
        }
    }

    return all_datastes


logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def calculate_quality(calc_automl: Callable, delete_dir: bool = True):

    # dataset_name = "used_cars_dataset"
    dataset_name = "buzz_dataset"

    config = copy(datasets()[dataset_name])
    config["use_algos"] = [["lgb"]]

    # seeds = [1, 42, 100, 200, 333, 555, 777, 2000, 50000, 100500,
    #              200000, 300000, 1_000_000, 2_000_000, 5_000_000, 74909, 54179, 68572, 25425]

    cv = 3
    seeds = [42]
    results = []
    for seed in seeds:
        cfg = deepcopy(config)
        cfg['seed'] = seed
        cfg['cv'] = cv

        os.environ[LOG_DATA_DIR] = f"./dumps/datalogs_{dataset_name}_{seed}"
        if os.path.exists(os.environ[LOG_DATA_DIR]) and delete_dir:
            shutil.rmtree(os.environ[LOG_DATA_DIR])

        log_config("general", cfg)

        res = calc_automl(**cfg)
        results.append(res)
        logger.info(f"Result for seed {seed}: {res}")

    mvals = [f"{r['metric_value']:_.2f}" for r in results]
    print("OOf on train metric")
    pprint(mvals)

    test_mvals = [f"{r['test_metric_value']:_.2f}" for r in results]
    print("Test metric")
    pprint(test_mvals)


if __name__ == "__main__":
    calculate_quality(lama_automl)
    calculate_quality(spark_automl, delete_dir=False)
