import logging.config
import logging.config
import os
import shutil
from copy import deepcopy, copy
from pprint import pprint
from typing import Callable

from dataset_utils import datasets
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT
from lightautoml.utils.tmp_utils import LOG_DATA_DIR, log_config
from spark_used_cars import calculate_automl as spark_automl

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def calculate_quality(calc_automl: Callable, delete_dir: bool = True):

    # dataset_name = "used_cars_dataset"
    dataset_name = "lama_test_dataset"
    # dataset_name = "buzz_dataset"

    config = copy(datasets()[dataset_name])
    config["use_algos"] = [["lgb_tuned"]]

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
    # calculate_quality(lama_automl)
    calculate_quality(spark_automl, delete_dir=False)
