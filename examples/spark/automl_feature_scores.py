import logging.config

from examples_utils import get_dataset
from examples_utils import get_spark_session
from examples_utils import prepare_test_and_train

from sparklightautoml.automl.presets.tabular_presets import SparkTabularAutoML
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import VERBOSE_LOGGING_FORMAT
from sparklightautoml.utils import log_exec_timer
from sparklightautoml.utils import logging_config


logging.config.dictConfig(logging_config(log_filename="/tmp/slama.log"))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


if __name__ == "__main__":
    spark = get_spark_session()

    seed = 42
    cv = 2
    use_algos = [["linear_l2"]]
    dataset_name = "lama_test_dataset"
    dataset = get_dataset(dataset_name)

    with log_exec_timer("spark-lama training") as train_timer:
        task = SparkTask(dataset.task_type)
        train_data, test_data = prepare_test_and_train(dataset, seed)

        test_data_dropped = test_data

        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            lgb_params={"use_single_dataset_mode": True},
            linear_l2_params={"default_params": {"regParam": [1]}},
            general_params={"use_algos": use_algos},
            reader_params={"cv": cv, "advanced_roles": False},
            tuning_params={"fit_on_holdout": True, "max_tuning_iter": 50, "max_tuning_time": 3600},
        )

        oof_predictions = automl.fit_predict(train_data, roles=dataset.roles).persist()

    logger.info("Predicting on out of fold")

    score = task.get_dataset_metric()
    metric_value = score(oof_predictions)

    logger.info(f"score for out-of-fold predictions: {metric_value}")

    feature_scores = automl.get_feature_scores(data=test_data_dropped, silent=False)

    print(feature_scores)

    oof_predictions.unpersist()
    # this is necessary if persistence_manager is of CompositeManager type
    # it may not be possible to obtain oof_predictions (predictions from fit_predict) after calling unpersist_all
    automl.persistence_manager.unpersist_all()
