from datetime import datetime

from decorator import contextmanager
from pyspark.sql import SparkSession


@contextmanager
def spark_session(parallelism: int = 1) -> SparkSession:
    spark = SparkSession.builder.config("master", f"local[{parallelism}]").getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


@contextmanager
def print_exec_time():
    start = datetime.now()
    yield
    end = datetime.now()
    duration = (end - start).total_seconds()
    print(f"Exec time: {duration}")