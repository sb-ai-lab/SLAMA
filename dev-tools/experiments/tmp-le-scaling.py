import logging.config

import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from examples_utils import get_spark_session
from lightautoml.dataset.roles import CategoryRole
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.transformers.categorical import SparkLabelEncoderEstimator
from lightautoml.spark.utils import log_exec_timer, logging_config, VERBOSE_LOGGING_FORMAT

import numpy as np

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def main(spark: SparkSession, path: str):
    execs = int(spark.conf.get('spark.executor.instances'))
    cores = int(spark.conf.get('spark.executor.cores'))

    df = spark.read.json(path).repartition(execs * cores).cache()
    df.write.mode('overwrite').format('noop').save()

    cat_roles = {
       c: CategoryRole(dtype=np.float32) for c in df.columns
    }

    with log_exec_timer("SparkLabelEncoder") as le_timer:
        estimator = SparkLabelEncoderEstimator(
            input_cols=list(cat_roles.keys()),
            input_roles=cat_roles
        )

        transformer = estimator.fit(df)

    with log_exec_timer("SparkLabelEncoder transform") as le_transform_timer:
        df = transformer.transform(df).cache()
        df.write.mode('overwrite').format('noop').save()
    # import time
    # time.sleep(600)

    return {
        "le_fit": le_timer.duration,
        "le_transform": le_transform_timer.duration
    }


if __name__ == "__main__":
    spark_sess = get_spark_session()
    # One can run:
    # 1. main(dataset_name="used_cars_dataset", seed=42)
    # 2. multirun(spark_sess, dataset_name="used_cars_dataset")
    main(spark_sess, path="/opt/spark_data/data_for_LE_TE_tests/1000000_rows_1000_columns_cardinality_10000_id.json")

    import time
    time.sleep(600)

    spark_sess.stop()
