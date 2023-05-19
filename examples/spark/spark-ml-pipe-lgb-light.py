import logging.config
import logging.config

from pyspark.ml import PipelineModel
from pyspark.sql import functions as sf

from examples.spark.examples_utils import check_columns
from examples_utils import get_persistence_manager
from examples_utils import get_spark_session, prepare_test_and_train, get_dataset_attrs
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
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
        sdataset = sreader.fit_read(train_df, roles=roles, persistence_manager=persistence_manager)

        iterator = SparkFoldsIterator(sdataset).convert_to_holdout_iterator()

        spark_ml_algo = SparkBoostLGBM(freeze_defaults=False, use_single_dataset_mode=True)
        spark_features_pipeline = SparkLGBSimpleFeatures()

        ml_pipe = SparkMLPipeline(
            ml_algos=[spark_ml_algo],
            pre_selection=None,
            features_pipeline=spark_features_pipeline,
            post_selection=None
        )

        oof_preds_ds = ml_pipe.fit_predict(iterator)
        oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"OOF score: {oof_score}")

        test_column = "some_external_column"
        test_df = test_df.withColumn(test_column, sf.lit(42.0))

        # 1. first way (LAMA API)
        test_sds = sreader.read(test_df, add_array_attrs=True)
        test_preds_ds = ml_pipe.predict(test_sds)

        test_score = score(test_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"Test score (#1 way): {test_score}")

        # 2. second way (Spark ML API, save-load-predict)
        transformer = PipelineModel(stages=[sreader.transformer(add_array_attrs=True), ml_pipe.transformer()])

        transformer.write().overwrite().save("/tmp/reader_and_spark_ml_pipe_lgb")

        pipeline_model = PipelineModel.load("/tmp/reader_and_spark_ml_pipe_lgb")
        test_pred_df = pipeline_model.transform(test_df)

        check_columns(test_df, test_pred_df)

        test_pred_df = test_pred_df.select(
            SparkDataset.ID_COLUMN,
            sf.col(roles['target']).alias('target'),
            sf.col(spark_ml_algo.prediction_feature).alias('prediction')
        )
        test_score = score(test_pred_df)
        logger.info(f"Test score (#2 way): {test_score}")

    logger.info("Finished")

    spark.stop()
