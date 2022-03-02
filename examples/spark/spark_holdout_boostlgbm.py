import os
from typing import cast

from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.ml_algo.base import SparkTabularMLAlgo
from lightautoml.spark.ml_algo.boost_lgbm import SparkBoostLGBM
from lightautoml.spark.utils import spark_session
from lightautoml.spark.validation.iterators import SparkHoldoutIterator, SparkFoldsIterator
from lightautoml.validation.np_iterators import FoldsIterator
from tests.spark.unit.dataset_utils import datasets, load_dump_if_exist

pipeline_name = 'lgbsimple_features'
config = datasets()["used_cars_dataset_head60k"]
# config = datasets()['used_cars_dataset_0125x']
# config = datasets()['used_cars_dataset']

cv = 5
checkpoint_dir = '/opt/test_checkpoints/feature_pipelines'
path = config['path']
task_name = config['task_type']
ds_name = os.path.basename(os.path.splitext(path)[0])

dump_train_path = os.path.join(checkpoint_dir, f"dump_{pipeline_name}_{ds_name}_{cv}_train.dump") \
    if checkpoint_dir is not None else None
dump_test_path = os.path.join(checkpoint_dir, f"dump_{pipeline_name}_{ds_name}_{cv}_test.dump") \
    if checkpoint_dir is not None else None

with spark_session(master='local[4]') as spark:
    spark.sparkContext.setLogLevel("INFO")
    train_res = load_dump_if_exist(spark, dump_train_path)
    test_res = load_dump_if_exist(spark, dump_test_path)
    if not train_res or not test_res:
        raise ValueError("Dataset should be processed with feature pipeline "
                         "and the corresponding dump should exist. Please, run corresponding non-quality test first.")
    dumped_train_ds, _ = train_res
    dumped_test_ds, _ = test_res

    test_ds = dumped_test_ds.to_pandas()

    # Process spark-based features with LAMA
    pds = dumped_train_ds.to_pandas()

    train_valid = FoldsIterator(pds)
    ml_algo = BoostLGBM()
    ml_algo, oof_pred = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)
    assert ml_algo is not None
    test_pred = ml_algo.predict(test_ds)
    score = train_valid.train.task.get_dataset_metric()
    lama_on_spark_oof_metric = score(oof_pred)
    lama_on_spark_test_metric = score(test_pred)
    print(f"LAMA oof: {lama_on_spark_oof_metric}. LAMA test: {lama_on_spark_test_metric}")

    # # train_valid = SparkHoldoutIterator(dumped_train_ds)
    # train_valid = SparkFoldsIterator(dumped_train_ds)
    # ml_algo = SparkBoostLGBM(freeze_defaults=False)
    # ml_algo, oof_pred = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)
    # ml_algo = cast(SparkTabularMLAlgo, ml_algo)
    # assert ml_algo is not None
    # test_pred = ml_algo.predict(dumped_test_ds)
    # score = train_valid.train.task.get_dataset_metric()
    # spark_based_oof_metric = score(oof_pred[:, ml_algo.prediction_feature])
    # spark_based_test_metric = score(test_pred[:, ml_algo.prediction_feature])
    #
    # print(f"Spark oof: {spark_based_oof_metric}. Spark test: {spark_based_test_metric}")
