import pickle
from typing import cast

import pytest
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split

from lightautoml.dataset.roles import CategoryRole
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.tasks import Task
from . import spark
import pandas as pd


@pytest.mark.parametrize("cv", [1, 5, 10])
# @pytest.mark.parametrize("cv", [1])
def test_spark_reader(spark: SparkSession, cv: int):
    def checks(sds: SparkDataset, check_target_and_folds: bool = True):
        # 1. it should have _id
        # 2. it should have target
        # 3. it should have roles for all columns
        if check_target_and_folds:
            assert sds.target_column not in sds.data.columns
            assert isinstance(sds.target, SparkDataFrame) \
                   and sds.target_column in sds.target.columns \
                   and SparkDataset.ID_COLUMN in sds.target.columns
        assert SparkDataset.ID_COLUMN in sds.data.columns
        assert set(sds.features).issubset(sds.roles.keys())
        assert all(f in sds.data.columns for f in sds.features)

        if check_target_and_folds:
            assert "folds" in sds.__dict__ and sds.folds
            assert isinstance(sds.folds, SparkDataFrame)
            folds_sdf = cast(SparkDataFrame, sds.folds)
            assert len(folds_sdf.columns) == 2
            assert SparkDataset.ID_COLUMN in folds_sdf.columns and sds.folds_column in folds_sdf.columns

    df = spark.read.csv("../../examples/data/sampled_app_train.csv", header=True)
    sreader = SparkToSparkReader(task=Task("binary"), cv=cv)

    sdataset = sreader.fit_read(df)
    checks(sdataset)

    sdataset = sreader.read(df)
    checks(sdataset, check_target_and_folds=False)


    # category_feats_and_roles = [(feat, role) for feat, role in sds.roles.items() if isinstance(role, CategoryRole)]
    # category_feats = [feat for feat, _ in category_feats_and_roles]
    # category_roles = {feat: role for feat, role in category_feats_and_roles}
    #
    # data = sds.data.select(*category_feats).toPandas()
    # target_data = sds.target.toPandas()
    # with open("unit/resources/datasets/dataset_after_reader_dump.pickle", "wb") as f:
    #     dmp = (data, category_feats, category_roles, target_data)
    #     pickle.dump(dmp, f)

    # assert len(sds.folds) == cv
    #
    # correct_folds = (
    #     (SparkDataset.ID_COLUMN in train.columns) and (SparkDataset.ID_COLUMN in val.columns)
    #     for train, val in sds.folds
    # )
    #
    # assert all(correct_folds), "ID_COLUMN should be presented everywhere"
