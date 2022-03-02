import logging.config

from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session, log_exec_time

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

        with log_exec_time():
            # data reading and converting to SparkDataset
            df = spark.read.csv("examples/data/tiny_used_cars_data.csv", header=True, escape="\"")
            task = SparkTask("reg")
            sreader = SparkToSparkReader(task=task, cv=3, advanced_roles=False)
            sdataset = sreader.fit_read(df, roles=roles)

        logger.info("Finished")
