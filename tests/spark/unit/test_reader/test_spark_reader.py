from typing import cast

import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.roles import CategoryRole
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.tasks import Task
from lightautoml.spark.tasks.base import Task as SparkTask
from . import spark
import pandas as pd


@pytest.mark.parametrize("cv", [1, 5, 10])
# @pytest.mark.parametrize("cv", [5])
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

    # path = "../../examples/data/sampled_app_train.csv"
    # task_type = "binary"

    path = "../../examples/data/tiny_used_cars_data.csv"
    task_type = "reg"
    roles = {
        "target": 'price',
        "drop": ["dealer_zip", "description", "listed_date",
                 "year", 'Unnamed: 0', '_c0',
                 'sp_id', 'sp_name', 'trimId',
                 'trim_name', 'major_options', 'main_picture_url',
                 'interior_color', 'exterior_color'],
        "numeric": ['latitude', 'longitude', 'mileage']
    }

    df = spark.read.csv(path, header=True, escape="\"")
    sreader = SparkToSparkReader(task=SparkTask(task_type), cv=cv)

    sdataset = sreader.fit_read(df, roles=roles)
    checks(sdataset)

    sdataset = sreader.read(df)
    checks(sdataset, check_target_and_folds=False)

    # comparing with Pandas
    pdf = pd.read_csv(path, dtype={
        'fleet': 'str', 'frame_damaged': 'str',
        'has_accidents': 'str', 'isCab': 'str',
        'is_cpo': 'str', 'is_new': 'str',
        'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str'
    })
    preader = PandasToPandasReader(task=Task(task_type), cv=cv)
    pdataset = preader.fit_read(pdf, roles=roles)

    sdiff = set(sdataset.features).symmetric_difference(pdataset.features)
    assert len(sdiff) == 0, f"Features sets are different: {sdiff}"

    feat_and_roles = [
        (feat, srole, pdataset.roles[feat])
        for feat, srole in sdataset.roles.items()
    ]

    not_equal_roles = [(feat, srole, prole) for feat, srole, prole in feat_and_roles if srole != prole]
    assert len(not_equal_roles) == 0, f"Roles are different: {not_equal_roles}"

    # two checks on CategoryRole to make PyCharm field resolution happy
    not_equal_encoding_types = [
        feat for feat, srole, prole in feat_and_roles
        if isinstance(srole, CategoryRole) and isinstance(prole, CategoryRole)
        and srole.encoding_type != prole.encoding_type
    ]
    assert len(not_equal_encoding_types) ==0 , f"Encoding types are different: {not_equal_encoding_types}"
