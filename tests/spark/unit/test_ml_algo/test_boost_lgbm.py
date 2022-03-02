import os
from typing import Dict, Any, cast

import pytest
from pyspark.sql import SparkSession

from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.spark.ml_algo.base import SparkTabularMLAlgo
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.validation.iterators import SparkHoldoutIterator
from .. import spark_with_deps
from ..dataset_utils import get_test_datasets, load_dump_if_exist

spark = spark_with_deps

CV = 5


@pytest.mark.parametrize("config,cv", [(ds, CV) for ds in get_test_datasets(setting="multiclass")])
def test_boostlgbm(spark: SparkSession, config: Dict[str, Any], cv: int):
    checkpoint_dir = '/opt/test_checkpoints/feature_pipelines'
    path = config['path']
    pipeline_name = 'lgbsimple_features'
    ds_name = os.path.basename(os.path.splitext(path)[0])

    dump_train_path = os.path.join(checkpoint_dir, f"dump_{pipeline_name}_{ds_name}_{cv}_train.dump") \
        if checkpoint_dir is not None else None
    dump_test_path = os.path.join(checkpoint_dir, f"dump_{pipeline_name}_{ds_name}_{cv}_test.dump") \
        if checkpoint_dir is not None else None

    train_res = load_dump_if_exist(spark, dump_train_path)
    test_res = load_dump_if_exist(spark, dump_test_path)
    if not train_res or not test_res:
        raise ValueError("Dataset should be processed with feature pipeline "
                         "and the corresponding dump should exist. Please, run corresponding non-quality test first.")

    dumped_train_ds, _ = train_res
    dumped_test_ds, _ = test_res

    train_valid = SparkHoldoutIterator(dumped_train_ds)
    ml_algo = SparkBoostLGBM(cacher_key='test')
    ml_algo, oof_pred = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)
    ml_algo = cast(SparkTabularMLAlgo, ml_algo)
    assert ml_algo is not None
    test_pred = ml_algo.predict(dumped_test_ds)
    score = train_valid.train.task.get_dataset_metric()
    spark_based_oof_metric = score(oof_pred[:, ml_algo.prediction_feature])
    spark_based_test_metric = score(test_pred[:, ml_algo.prediction_feature])


# def test_smoke_boost_lgbm_v2(spark: SparkSession):
#
#     with open("unit/resources/datasets/dump_tabular_automl_lgb_cb_linear/Lvl_0_Pipe_0_apply_selector.pickle", "rb") as f:
#         data, target, features, roles = pickle.load(f)
#
#     nds = NumpyDataset(data[4000:, :], features, roles, task=Task("binary"))
#     pds = nds.to_pandas()
#     target = pd.Series(target[4000:])
#
#     sds = from_pandas_to_spark(pds, spark, target)
#     iterator = DummyIterator(train=sds)
#
#     ml_algo = SparkBoostLGBM()
#     pred_ds = ml_algo.fit_predict(iterator)
#
#     predicted_sdf = pred_ds.data
#     ppdf = predicted_sdf.toPandas()
#
#     assert SparkDataset.ID_COLUMN in predicted_sdf.columns
#     assert len(pred_ds.features) == 1
#     assert pred_ds.features[0].endswith("_prediction")
#     assert pred_ds.features[0] in predicted_sdf.columns


