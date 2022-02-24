import os
from typing import cast

import pandas as pd

from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.linear_sklearn import LinearLBFGS
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.spark.ml_algo.base import SparkTabularMLAlgo
from lightautoml.spark.ml_algo.linear_pyspark import SparkLinearLBFGS
from lightautoml.spark.utils import spark_session
from lightautoml.spark.validation.iterators import SparkFoldsIterator
from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import FoldsIterator
from tests.spark.unit.dataset_utils import get_test_datasets, load_dump_if_exist

PIPELINE_NAME = 'linear_features'
CV = 3

configs = get_test_datasets(setting="multiclass")
config = configs[1]

checkpoint_dir = '/opt/test_checkpoints/feature_pipelines'
path = config['path']
ds_name = os.path.basename(os.path.splitext(path)[0])

dump_train_path = os.path.join(checkpoint_dir, f"dump_{PIPELINE_NAME}_{ds_name}_{CV}_train.dump") \
    if checkpoint_dir is not None else None
dump_test_path = os.path.join(checkpoint_dir, f"dump_{PIPELINE_NAME}_{ds_name}_{CV}_test.dump") \
    if checkpoint_dir is not None else None


ml_alg_kwargs = {}

fp_lama_clazz = LinearFeatures
ml_algo_lama_clazz = LinearLBFGS
ml_algo_spark_clazz = SparkLinearLBFGS

with spark_session(master="local[4]") as spark:
    train_res = load_dump_if_exist(spark, dump_train_path)
    test_res = load_dump_if_exist(spark, dump_test_path)
    if not train_res or not test_res:
        raise ValueError("Dataset should be processed with feature pipeline "
                         "and the corresponding dump should exist. Please, run corresponding non-quality test first.")
    dumped_train_ds, _ = train_res
    dumped_test_ds, _ = test_res

    test_ds = dumped_test_ds.to_pandas() if ml_algo_lama_clazz == BoostLGBM else dumped_test_ds.to_pandas().to_numpy()

    # Process spark-based features with LAMA
    pds = dumped_train_ds.to_pandas() if ml_algo_lama_clazz == BoostLGBM else dumped_train_ds.to_pandas().to_numpy()

    # compare with native features of LAMA
    print("=========================LAMA==================================")
    train_valid = FoldsIterator(pds)
    read_csv_args = {'dtype': config['dtype']} if 'dtype' in config else dict()
    train_pdf = pd.read_csv(config['train_path'], **read_csv_args)
    test_pdf = pd.read_csv(config['test_path'], **read_csv_args)
    # train_pdf, test_pdf = train_test_split(pdf, test_size=0.2, random_state=100)
    reader = PandasToPandasReader(task=Task(train_valid.train.task.name), cv=CV, advanced_roles=False)
    train_ds = reader.fit_read(train_pdf, roles=config['roles'])
    test_ds_2 = reader.read(test_pdf, add_array_attrs=True)
    lama_pipeline = fp_lama_clazz(**ml_alg_kwargs)
    lama_feats = lama_pipeline.fit_transform(train_ds)
    lama_test_feats = lama_pipeline.transform(test_ds_2)
    lama_feats = lama_feats if ml_algo_lama_clazz == BoostLGBM else lama_feats.to_numpy()
    train_valid = FoldsIterator(lama_feats.to_numpy())
    ml_algo = ml_algo_lama_clazz()
    ml_algo, oof_pred = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)
    assert ml_algo is not None
    test_pred = ml_algo.predict(lama_test_feats)
    # test_pred = ml_algo.predict(test_ds)
    score = train_valid.train.task.get_dataset_metric()
    lama_oof_metric = score(oof_pred)
    lama_test_metric = score(test_pred)

    # LAMA-on-spark features
    print("=========================LAMA-on-spark-features==================================")
    train_valid = FoldsIterator(pds)
    ml_algo = ml_algo_lama_clazz()
    ml_algo, oof_pred = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)
    assert ml_algo is not None
    test_pred = ml_algo.predict(test_ds)
    score = train_valid.train.task.get_dataset_metric()
    lama_on_spark_oof_metric = score(oof_pred)
    lama_on_spark_test_metric = score(test_pred)

    print("=========================Spark==================================")
    train_valid = SparkFoldsIterator(dumped_train_ds)
    ml_algo = ml_algo_spark_clazz()
    ml_algo, oof_pred = tune_and_fit_predict(ml_algo, DefaultTuner(), train_valid)
    ml_algo = cast(SparkTabularMLAlgo, ml_algo)
    assert ml_algo is not None
    test_pred = ml_algo.predict(dumped_test_ds)
    score = train_valid.train.task.get_dataset_metric()
    spark_based_oof_metric = score(oof_pred[:, ml_algo.prediction_feature])
    spark_based_test_metric = score(test_pred[:, ml_algo.prediction_feature])

    print(f"LAMA oof: {lama_oof_metric}. LAMA test: {lama_test_metric}.")
    print(f"LAMA-on-spark oof: {lama_on_spark_oof_metric}. LAMA-on-spark test: {lama_on_spark_test_metric}.")
    print(f"Spark oof: {spark_based_oof_metric}. LAMA-on-spark test: {spark_based_test_metric}.")

    max_diff_in_percents = 0.05

    # assert spark_based_test_metric > lama_test_metric or abs(
    #     (lama_test_metric - spark_based_test_metric) / max(lama_test_metric,
    #                                                        spark_based_test_metric)) < max_diff_in_percents
    # assert spark_based_test_metric > lama_test_metric or abs(
    #     (lama_test_metric - spark_based_test_metric) / min(lama_test_metric,
    #                                                        spark_based_test_metric)) < max_diff_in_percents
