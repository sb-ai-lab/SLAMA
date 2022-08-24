#!/usr/bin/python3

import time

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

result = spark.sparkContext.parallelize([i for i in range(10)]).sum()
print(f"Test result: {result}")

spark.stop()
