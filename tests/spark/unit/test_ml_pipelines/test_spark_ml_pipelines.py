from copy import copy

from pyspark.sql import SparkSession

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures, SparkLGBAdvancedPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from .. import from_pandas_to_spark, spark_with_deps

spark = spark_with_deps


def test_lgb_simple_pipeline(spark: SparkSession):
    # path = "./examples/data/sampled_app_train.csv"
    path = "./examples/data/tiny_used_cars_data.csv"
    task_type = "binary"
    # roles = {"target": "TARGET", "drop": ["SK_ID_CURR"]}
    roles = {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            }

    df = spark.read.csv(path, header=True, escape="\"")
    sreader = SparkToSparkReader(task=SparkTask(task_type), cv=2, advanced_roles=False)
    sdataset = sreader.fit_read(df, roles=roles)

    fp = SparkLGBSimpleFeatures(sdataset.roles)
    out_ds = fp.fit_transform(sdataset)
    transformer = fp.transformer

    out_df = transformer.transform(sdataset.data)
    out_df.write.mode('overwrite').format('noop').save()


def test_lgb_advanced_pipeline(spark: SparkSession):
    # path = "./examples/data/sampled_app_train.csv"
    path = "./examples/data/tiny_used_cars_data.csv"
    task_type = "binary"
    # roles = {"target": "TARGET", "drop": ["SK_ID_CURR"]}
    roles = {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            }

    df = spark.read.csv(path, header=True, escape="\"")
    sreader = SparkToSparkReader(task=SparkTask(task_type), cv=2, advanced_roles=False)
    sdataset = sreader.fit_read(df, roles=roles)

    # features = copy(sdataset.features)
    # roles = copy(sdataset.roles)

    sdataset.data = (
        sdataset
        .data
        .join(sdataset.target, on=SparkDataset.ID_COLUMN)
        .join(sdataset.folds, on=SparkDataset.ID_COLUMN)
    )

    fp = SparkLGBAdvancedPipeline(sdataset.roles)
    out_ds = fp.fit_transform(sdataset)
    transformer = fp.transformer

    out_df = transformer.transform(sdataset.data)
    out_df.write.mode('overwrite').format('noop').save()
