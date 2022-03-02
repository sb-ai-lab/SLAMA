import logging.config
import logging.config
from copy import deepcopy

from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.spark.utils import logging_config, VERBOSE_LOGGING_FORMAT, spark_session, log_exec_time
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

        with log_exec_time():
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

            cacher_key = "main_cache"

            iterator = SparkFoldsIterator(sdataset, n_folds=3)

            spark_ml_algo = SparkBoostLGBM(cacher_key='example', freeze_defaults=False)
            spark_features_pipeline = SparkLGBAdvancedPipeline(cacher_key=cacher_key, **ml_alg_kwargs)
            spark_selector = ImportanceCutoffSelector(
                cutoff=0.0,
                feature_pipeline=SparkLGBSimpleFeatures(cacher_key='preselector'),
                ml_algo=SparkBoostLGBM(cacher_key='example', freeze_defaults=False),
                imp_estimator=ModelBasedImportanceEstimator()
            )

            ml_pipe = SparkMLPipeline(
                cacher_key=cacher_key,
                ml_algos=[spark_ml_algo],
                pre_selection=spark_selector,
                features_pipeline=spark_features_pipeline,
                post_selection=None
            )

            _ = ml_pipe.fit_predict(iterator)

            final_result = ml_pipe.transformer.transform(sdataset.data)
            final_result.write.mode('overwrite').format('noop').save()

        logger.info("Finished")
