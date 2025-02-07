import os
from pyspark.sql import SparkSession

(
    SparkSession.builder
    .config(
        "spark.jars.packages",
        f"com.microsoft.azure:synapseml_2.12:{os.environ['SYNAPSEML_VER']}"
    )
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
    .getOrCreate()
)