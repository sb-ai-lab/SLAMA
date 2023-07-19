import logging.config

import pytest
from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import PersistenceManager, PersistenceLevel
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager, LocalCheckpointPersistenceManager, \
    BucketedPersistenceManager, CompositePersistenceManager
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT
from .utils import DummyTabularAutoML
from .. import make_spark, spark as spark_sess, BUCKET_NUMS

make_spark = make_spark
spark = spark_sess

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/lama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


# noinspection PyShadowingNames
@pytest.mark.parametrize("persistence_manager", [
    PlainCachePersistenceManager(),
    LocalCheckpointPersistenceManager(),
    BucketedPersistenceManager(bucketed_datasets_folder="/tmp", bucket_nums=10),
    CompositePersistenceManager({
        PersistenceLevel.READER: BucketedPersistenceManager(bucketed_datasets_folder="/tmp", bucket_nums=10),
        PersistenceLevel.REGULAR: PlainCachePersistenceManager(),
        PersistenceLevel.CHECKPOINT: BucketedPersistenceManager(bucketed_datasets_folder="/tmp", bucket_nums=10)
    }),
    CompositePersistenceManager({
        PersistenceLevel.READER: BucketedPersistenceManager(
            bucketed_datasets_folder="/tmp", bucket_nums=BUCKET_NUMS, no_unpersisting=True
        ),
        PersistenceLevel.REGULAR: PlainCachePersistenceManager(prune_history=False),
        PersistenceLevel.CHECKPOINT: PlainCachePersistenceManager(prune_history=False)
    }),
    CompositePersistenceManager({
        PersistenceLevel.READER: BucketedPersistenceManager(
            bucketed_datasets_folder="/tmp", bucket_nums=BUCKET_NUMS, no_unpersisting=True
        ),
        PersistenceLevel.REGULAR: PlainCachePersistenceManager(prune_history=False),
        PersistenceLevel.CHECKPOINT: BucketedPersistenceManager(
            bucketed_datasets_folder="/tmp", bucket_nums=BUCKET_NUMS
        ),
    })
])
def test_automl_preset(spark: SparkSession, persistence_manager: PersistenceManager):
    n_classes = 10

    train_data = spark.createDataFrame([
        {"a": i, "b": 100 + i, "c": 100 * i, "TARGET": i % n_classes} for i in range(120)
    ])

    test_data = spark.createDataFrame([
        {"a": i, "b": 100 + i, "c": 100 * i, "TARGET": i % n_classes} for i in range(120, 140)
    ])

    automl = DummyTabularAutoML(n_classes=n_classes)

    # 1. check for output result, features, roles (required columns in data, including return_all_predictions)
    # 2. checking for layer-to-layer data transfer (internal in DummyTabularAutoML):
    #   - all predictions of the first level are available in all pipes of the second level
    #   - all inputs data are presented in all pipes of the first level
    #   - all inputs data are presented in all pipes of the second level (if skip_conn)
    # 3. blending and return_all_predictions works correctly
    oof_ds = automl.fit_predict(
        train_data,
        roles={"target": "TARGET"},
        persistence_manager=persistence_manager
    ).persist()

    logger.info("Starting to predict")

    pred_ds = automl.predict(test_data, persistence_manager=persistence_manager).persist()

    assert len(persistence_manager.children) == 0
    assert len(persistence_manager.all_datasets) == 2

    oof_ds.unpersist()
    pred_ds.unpersist()

    assert len(persistence_manager.all_datasets) == 0

    # automl_model_path = "/tmp/slama_test.model"
    # automl.transformer().write().overwrite().save(automl_model_path)
    # pipeline_model = PipelineModel.load(automl_model_path)
    #
    # te_pred = pipeline_model.transform(test_data)
    # te_pred.write.mode('overwrite').format('noop').save()

    logger.info("Finished")
