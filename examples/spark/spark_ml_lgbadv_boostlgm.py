import logging.config
import logging.config
from copy import deepcopy
from typing import cast

from pyspark.ml import PipelineModel

from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session
from lightautoml.spark.validation.iterators import SparkFoldsIterator

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

        iterator = SparkFoldsIterator(sdataset, n_folds=3)
        lgb_features = SparkLGBAdvancedPipeline(**ml_alg_kwargs)
        spark_ml_algo = SparkBoostLGBM(cacher_key='example', freeze_defaults=False)

        # # Process features and train the model
        iterator = iterator.apply_feature_pipeline(lgb_features)
        spark_ml_algo, _ = tune_and_fit_predict(spark_ml_algo, DefaultTuner(), iterator)
        spark_ml_algo = cast(SparkBoostLGBM, spark_ml_algo)

        final = PipelineModel(stages=[lgb_features.transformer, spark_ml_algo.transformer])

        final_result = final.transform(sdataset.data)
        final_result.write.mode('overwrite').format('noop').save()

        logger.info("Finished")
