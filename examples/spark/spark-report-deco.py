import logging.config
import os
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataFrame
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.report import SparkReportDeco
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import VERBOSE_LOGGING_FORMAT
from lightautoml.spark.utils import log_exec_timer
from lightautoml.spark.utils import logging_config

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def prepare_test_and_train(spark: SparkSession, path: str, seed: int) -> Tuple[SparkDataFrame, SparkDataFrame]:
    data = spark.read.csv(path, header=True, escape="\"")

    data = data.select(
        '*',
        F.monotonically_increasing_id().alias(SparkDataset.ID_COLUMN),
        F.rand(seed).alias('is_test')
    ).cache()
    data.write.mode('overwrite').format('noop').save()

    train_data = data.where(F.col('is_test') < 0.8).drop('is_test').cache()
    test_data = data.where(F.col('is_test') >= 0.8).drop('is_test').cache()

    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    return train_data, test_data


def get_spark_session():
    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        return SparkSession.builder.getOrCreate()

    spark_sess = (
        SparkSession
        .builder
        .master("local[5]")
        .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.jar")
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "12g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    return spark_sess


if __name__ == "__main__":
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    seed = 42
    cv = 2

    use_algos = [["lgb"]]
    task_type = "binary"
    roles = {"target": "TARGET", "drop": ["SK_ID_CURR"]}

    DATASET_DIR = '/tmp/'
    DATASET_NAME = 'sampled_app_train.csv'
    DATASET_FULLNAME = os.path.join(DATASET_DIR, DATASET_NAME)
    DATASET_URL = 'https://raw.githubusercontent.com/sberbank-ai-lab/LightAutoML/master/examples/data/sampled_app_train.csv'

    if not os.path.exists(DATASET_FULLNAME):
        os.makedirs(DATASET_DIR, exist_ok=True)

        dataset = requests.get(DATASET_URL).text
        with open(DATASET_FULLNAME, 'w') as output:
            output.write(dataset)

    data = pd.read_csv(DATASET_FULLNAME)
    data['EMP_DATE'] = (np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
                        ).astype(str)

    data.to_csv("/tmp/sampled_app_train.csv", index=False)

    train_data, test_data = prepare_test_and_train(spark, "/tmp/sampled_app_train.csv", seed)

    with log_exec_timer("spark-lama training") as train_timer:
        task = SparkTask(task_type)

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            lgb_params={'use_single_dataset_mode': True, "default_params": {"numIterations": 3000}},
            linear_l2_params={"default_params": {"regParam": [1]}},
            general_params={"use_algos": use_algos},
            reader_params={"cv": cv, "advanced_roles": False, 'random_state': seed}
        )

        report_automl = SparkReportDeco(
            output_path="/tmp/spark",
            report_file_name="spark_lama_report.html",
            interpretation=True
        )(automl)

        report_automl.fit_predict(train_data, roles=roles)
        report_automl.predict(test_data, add_reader_attrs=True)
