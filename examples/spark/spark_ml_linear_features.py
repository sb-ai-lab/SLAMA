import logging.config
import logging.config
from copy import deepcopy

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.features.linear_pipeline import SparkLinearFeatures
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    with spark_session(master="local[4]") as spark:
        roles = {
            "target": 'price',
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            "numeric": ['latitude', 'longitude', 'mileage']
        }

        # data reading and converting to SparkDataset
        df = spark.read.csv("examples/data/tiny_used_cars_data.csv", header=True, escape="\"")
        task = SparkTask("reg")
        sreader = SparkToSparkReader(task=task, cv=3, advanced_roles=False)
        sdataset = sreader.fit_read(df, roles=roles)

        ml_alg_kwargs = {
            'auto_unique_co': 10,
            'max_intersection_depth': 3,
            'multiclass_te_co': 3,
            'output_categories': True,
            'top_intersections': 4
        }

        # # Spark ML pipeline
        simple_pipline_builder = SparkLinearFeatures(**ml_alg_kwargs)
        sdataset_feats = simple_pipline_builder.fit_transform(sdataset)
        sdataset_feats.data.toPandas().to_csv("/tmp/after_linear_features.csv")

        

        logger.info("Finished")
