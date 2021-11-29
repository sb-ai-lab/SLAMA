from decorator import contextmanager
from pyspark.sql import SparkSession


@contextmanager
def spark_session(parallelism: int = 1) -> SparkSession:
    spark = SparkSession.builder.config("master", f"local[{parallelism}]").getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()