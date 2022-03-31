import logging.config

import pytest
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from examples_utils import get_spark_session, prepare_test_and_train, get_dataset_attrs
from lightautoml.spark.automl.presets.tabular_presets import SparkTabularAutoML
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.tasks.base import SparkTask
from lightautoml.spark.utils import VERBOSE_LOGGING_FORMAT
from lightautoml.spark.utils import log_exec_timer
from lightautoml.spark.utils import logging_config

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    spark = get_spark_session()

    seed = 42
    cv = 2
    use_algos = [["lgb"]]
    dataset_name = "ipums_97"
    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)


    train_data, test_data = prepare_test_and_train(spark, path, seed)

    with log_exec_timer("spark-lama training") as train_timer:
        task = SparkTask(task_type)

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            lgb_params={'use_single_dataset_mode': True},
            general_params={"use_algos": use_algos},
            reader_params={"cv": cv, "advanced_roles": False, 'random_state': seed},
            tuning_params={'fit_on_holdout': True, 'max_tuning_iter': 10, 'max_tuning_time': 3600}
        )

        preds = automl.fit_predict(train_data, roles)

    transformer = automl.make_transformer()
    transformer.write().overwrite().save("hdfs://namenode:9000/automl_multiclass")

    with log_exec_timer("spark-lama predicting on test") as predict_timer_2:
        te_pred = transformer.transform(test_data)

        pred_column = next(c for c in te_pred.columns if c.startswith('prediction'))
        score = task.get_dataset_metric()
        expected_metric_value = score(te_pred.select(
            SparkDataset.ID_COLUMN,
            F.col(roles['target']).alias('target'),
            F.col(pred_column).alias('prediction')
        ))

        logger.info(f"score for test predictions: {expected_metric_value}")

    with log_exec_timer("spark-lama predicting on test via loaded pipeline") as predict_timer_3:
        pipeline_model = PipelineModel.load("hdfs://namenode:9000/automl_multiclass")
        te_pred = pipeline_model.transform(test_data)

        pred_column = next(c for c in te_pred.columns if c.startswith('prediction'))
        score = task.get_dataset_metric()
        actual_metric_value = score(te_pred.select(
            SparkDataset.ID_COLUMN,
            F.col(roles['target']).alias('target'),
            F.col(pred_column).alias('prediction')
        ))
        logger.info(f"score for test predictions via loaded pipeline: {actual_metric_value}")

    assert expected_metric_value == pytest.approx(actual_metric_value, 0.1)
