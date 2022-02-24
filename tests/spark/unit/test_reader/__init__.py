import os

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
# @pytest.fixture(scope="function")
def spark() -> SparkSession:
    os.environ['PYSPARK_PYTHON'] = '/home/nikolay/.conda/envs/LAMA/bin/python'

    spark = SparkSession.builder.config("master", "local[4]").config('spark.driver.memory', '8g').getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()
