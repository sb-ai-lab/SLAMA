import logging.config

import pyspark.sql.functions as sf
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from pyspark.ml import PipelineModel

from examples_utils import get_persistence_manager
from examples_utils import get_spark_session, get_dataset, prepare_test_and_train
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.pipelines.selection.base import BugFixSelectionPipelineWrapper
from sparklightautoml.pipelines.selection.base import SparkSelectionPipelineWrapper
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
        sdataset = sreader.fit_read(train_df, roles=dataset.roles, persistence_manager=persistence_manager)

        iterator = SparkFoldsIterator(sdataset, n_folds=cv)

        spark_ml_algo = SparkLinearLBFGS(default_params={'regParam': [1e-5]})
        spark_features_pipeline = SparkLinearFeatures(**ml_alg_kwargs)
        spark_selector = BugFixSelectionPipelineWrapper(ImportanceCutoffSelector(
            cutoff=0.0,
            feature_pipeline=SparkLGBSimpleFeatures(),
            ml_algo=SparkBoostLGBM(freeze_defaults=False),
            imp_estimator=ModelBasedImportanceEstimator()
        ))

        ml_pipe = SparkMLPipeline(
            ml_algos=[spark_ml_algo],
            pre_selection=SparkSelectionPipelineWrapper(spark_selector),
            features_pipeline=spark_features_pipeline,
            post_selection=None
        )

        oof_preds_ds = ml_pipe.fit_predict(iterator).persist()
        oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"OOF score: {oof_score}")

        # 1. first way (LAMA API)
        test_sds = sreader.read(test_df, add_array_attrs=True)
        test_preds_ds = ml_pipe.predict(test_sds)
        test_score = score(test_preds_ds[:, spark_ml_algo.prediction_feature])
        logger.info(f"Test score (#1 way): {test_score}")

        # 2. second way (Spark ML API)
        transformer = PipelineModel(stages=[sreader.transformer(add_array_attrs=True), ml_pipe.transformer()])
        test_pred_df = transformer.transform(test_df)
        test_pred_df = test_pred_df.select(
            SparkDataset.ID_COLUMN,
            sf.col(dataset.roles['target']).alias('target'),
            sf.col(spark_ml_algo.prediction_feature).alias('prediction')
        )
        test_score = score(test_pred_df)
        logger.info(f"Test score (#2 way): {test_score}")

    logger.info("Finished")

    oof_preds_ds.unpersist()
    # this is necessary if persistence_manager is of CompositeManager type
    # it may not be possible to obtain oof_predictions (predictions from fit_predict) after calling unpersist_all
    persistence_manager.unpersist_all()

    spark.stop()
