import logging.config
import os
from typing import List

import numpy as np
import pytest
from lightautoml.dataset.roles import NumericRole
from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import BucketedPersistenceManager
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, get_current_session
from .. import BUCKET_NUMS, spark_for_function, spark_hdfs as spark_hdfs_sess, HDFS_TMP_SLAMA_DIR

spark = spark_for_function
spark_hdfs = spark_hdfs_sess

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def do_test_bucketed_persistence_manager(spark_session: SparkSession):
    def make_ds(name: str) -> SparkDataset:
        data = [
            [0, 0, 42, 1, 1, 1],
            [1, 0, 43, 2, 1, 3],
            [2, 1, 44, 1, 2, 3],
            [3, 1, 45, 1, 2, 2],
            [4, 2, 46, 3, 1, 1],
            [5, 2, 47, 4, 1, 2],
        ]

        in_cols = ["_id", "fold", "seed", "a", "b", "c"]

        roles = {col: NumericRole(np.int32) for col in in_cols}

        df_data = [
            {col: val for col, val in zip(in_cols, row)}
            for row in data
        ]
        df = spark_session.createDataFrame(df_data)

        return SparkDataset(df, roles, name=name)

    def existing_tables() -> List[str]:
        return [
            table.name.split("_")[0]
            for table in get_current_session().catalog.listTables()
        ]

    pmanager = BucketedPersistenceManager(
        bucketed_datasets_folder=f"hdfs://node21.bdcl:9000{HDFS_TMP_SLAMA_DIR}",
        bucket_nums=BUCKET_NUMS
    )

    # we lowercase all tables name in asserts due to spark creates tables only only with preprocessed names
    # that includes lowercasing
    # but Spark still able to drop table by the name in its original registry
    # so using the original name with upper cased letter doesn't present any problem for operations with tables
    # for instance, .sql("DROP TABLE datasetA") will work as exepected
    tables = ["datasetA", "datasetB"]
    dss = [pmanager.persist(make_ds(table)) for table in tables]

    assert set(existing_tables()) == set([table.lower() for table in tables])

    pmanager.unpersist(dss[0].uid)
    assert set(existing_tables()) == {"datasetB".lower()}

    pmanager.unpersist_all()
    assert set(existing_tables()) == set()


def test_bucketed_persistence_manager(spark: SparkSession):
    do_test_bucketed_persistence_manager(spark)


@pytest.mark.skipif(
    "DO_HDFS_BASED_TESTS" not in os.environ,
    reason="Env var 'DO_HDFS_BASED_TESTS' is not set. Not sure if the test may succeed."
)
def test_bucketed_persistence_manager_with_hdfs(spark_hdfs: SparkSession):
    """
        This test requires configured access to HDFS
        which may not be available for every enviroment
        env var 'DO_HDFS_BASED_TESTS' signals that the environment is propely configured and is ready to run this test
        otherwise we skip it
    """
    do_test_bucketed_persistence_manager(spark_hdfs)
