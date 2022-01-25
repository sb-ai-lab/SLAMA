import logging
import logging.config
from pprint import pprint
from typing import Any, Callable
from lama_used_cars import calculate_automl as lama_automl
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT
from spark_used_cars import calculate_automl as spark_automl


logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def calculate_quality(calc_automl: Callable[[str, int, Any], dict]):
    dataset_path = "examples/data/tiny_used_cars_data.csv"
    # dataset_path = "examples/data/small_used_cars_data.csv"
    # dataset_path = "/opt/0125x_cleaned.csv"
    use_algos = ("lgb", "linear_l2")
    # seeds = [1, 42, 100, 200, 333, 555, 777, 2000, 50000, 100500,
    #              200000, 300000, 1_000_000, 2_000_000, 5_000_000]
    seeds = [42, 100, 200]
    results = [calc_automl(dataset_path, seed, use_algos) for seed in seeds]

    mvals = [f"{r['metric_value']:_.2f}" for r in results]
    print("OOf on train metric")
    pprint(mvals)

    test_mvals = [f"{r['test_metric_value']:_.2f}" for r in results]
    print("Test metric")
    pprint(test_mvals)


if __name__ == "__main__":
    calculate_quality(lama_automl)
    # calculate_quality(spark_automl)
