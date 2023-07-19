import math
from typing import Dict, Any, cast

import pytest
from lightautoml.dataset.base import RolesDict
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.validation.iterators import SparkFoldsIterator
from .. import make_spark, spark as spark_sess
from ..dataset_utils import get_test_datasets

make_spark = make_spark
spark = spark_sess


@pytest.mark.skip(reason="СlassCastException in synapseml lightgbm due to unknown reason")
@pytest.mark.parametrize("config,cv", [(ds, 3) for ds in get_test_datasets(dataset="used_cars_dataset")])
def test_boost_lgb_oof_preds(spark: SparkSession, config: Dict[str, Any], cv: int):
    task_type = cast(str, config['task_type'])
    roles = cast(RolesDict, config['roles'])
    train_path = cast(str, config['train_path'])

    task = SparkTask(task_type)
    persistence_manager = PlainCachePersistenceManager()

    train_df = spark.read.csv(train_path, header=True, escape="\"")

    max_val_size = math.floor(train_df.count() * 0.1)

    sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
    spark_ml_algo = SparkBoostLGBM(
        freeze_defaults=False,
        use_single_dataset_mode=False,
        max_validation_size=max_val_size
    )
    spark_features_pipeline = SparkLGBSimpleFeatures()

    sdataset = sreader.fit_read(train_df, roles=roles, persistence_manager=persistence_manager)
    sdataset = spark_features_pipeline.fit_transform(sdataset)

    iterator = SparkFoldsIterator(sdataset)

    oof_preds = spark_ml_algo.fit_predict(iterator)

    null_preds_count = oof_preds.data.where(sf.isnull(spark_ml_algo.prediction_feature)).count()
    nan_preds_count = oof_preds.data.where(sf.isnan(spark_ml_algo.prediction_feature)).count()

    assert null_preds_count == 0 and nan_preds_count == 0
