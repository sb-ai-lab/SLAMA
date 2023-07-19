import logging.config

from examples_utils import get_persistence_manager
from examples_utils import get_spark_session, get_dataset, prepare_test_and_train
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
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
    persistence_manager = get_persistence_manager()

    seed = 42
    cv = 3
    dataset_name = "lama_test_dataset"
    dataset = get_dataset(dataset_name)

    ml_alg_kwargs = {
        'auto_unique_co': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': True,
        'top_intersections': 4
    }

    with log_exec_time():
        train_df, test_df = prepare_test_and_train(dataset, seed)

        task = SparkTask(dataset.task_type)
        score = task.get_dataset_metric()

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)

        spark_features_pipeline = SparkLinearFeatures(**ml_alg_kwargs)
        spark_ml_algo = SparkLinearLBFGS(default_params={'regParam': [1e-5]})

        sdataset = sreader.fit_read(train_df, roles=dataset.roles, persistence_manager=persistence_manager)
        sdataset = spark_features_pipeline.fit_transform(sdataset)
        iterator = SparkFoldsIterator(sdataset, n_folds=cv)
        oof_preds_ds = spark_ml_algo.fit_predict(iterator)

        oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"OOF score: {oof_score}")

        test_sds = sreader.read(test_df, add_array_attrs=True)
        test_sds = spark_features_pipeline.transform(test_sds)
        test_preds_ds = spark_ml_algo.predict(test_sds)

        test_score = score(test_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"Test score (before saving the intermediate dataset): {test_score}")

    logger.info("Finished")

    spark.stop()
