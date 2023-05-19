import logging.config
import logging.config

from examples.spark.examples_utils import FSOps
from examples_utils import get_persistence_manager
from examples_utils import get_spark_session, prepare_test_and_train, get_dataset_attrs
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask as SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_time
from sparklightautoml.validation.iterators import SparkFoldsIterator

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


if __name__ == "__main__":
    spark = get_spark_session()

    FSOps.get_fs("/tmp/just_a_test")

    seed = 42
    cv = 5
    dataset_name = "lama_test_dataset"
    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    persistence_manager = get_persistence_manager()

    ml_alg_kwargs = {
        'auto_unique_co': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': True,
        'top_intersections': 4
    }

    with log_exec_time():
        train_df, test_df = prepare_test_and_train(spark, path, seed)

        task = SparkTask(task_type)
        score = task.get_dataset_metric()

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        spark_ml_algo = SparkBoostLGBM(freeze_defaults=False, use_single_dataset_mode=False)
        spark_features_pipeline = SparkLGBSimpleFeatures()

        sdataset = sreader.fit_read(train_df, roles=roles, persistence_manager=persistence_manager)
        sdataset = spark_features_pipeline.fit_transform(sdataset)
        iterator = SparkFoldsIterator(sdataset).convert_to_holdout_iterator()
        oof_preds_ds = spark_ml_algo.fit_predict(iterator)

        oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"OOF score: {oof_score}")

        test_sds = sreader.read(test_df, add_array_attrs=True)
        test_sds = spark_features_pipeline.transform(test_sds)
        test_preds_ds = spark_ml_algo.predict(test_sds)

        test_score = score(test_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"Test score (before saving the intermediate dataset): {test_score}")

        test_sds.save("/tmp/test_sds.dataset", save_mode='overwrite')
        test_sds_2 = SparkDataset.load("/tmp/test_sds.dataset")

        test_preds_ds_2 = spark_ml_algo.predict(test_sds_2)

        test_score_2 = score(test_preds_ds_2[:, spark_ml_algo.prediction_feature])
        logger.info(f"Test score (after loading the intermediate dataset): {test_score_2}")

    logger.info("Finished")

    spark.stop()
