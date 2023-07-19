from typing import Dict, Any

import pandas as pd
import pytest
from lightautoml.dataset.roles import CategoryRole
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task
from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask as SparkTask
from .. import make_spark, spark as spark_sess
from ..dataset_utils import get_test_datasets

make_spark = make_spark
spark = spark_sess


# noinspection PyShadowingNames
@pytest.mark.parametrize("config,cv,agr", [(ds, 5, False) for ds in get_test_datasets(setting="all-tasks")])
def test_spark_reader(spark: SparkSession, config: Dict[str, Any], cv: int, agr: bool):
    def checks(sds: SparkDataset, check_target_and_folds: bool = True):
        # 1. it should have _id
        # 2. it should have target
        # 3. it should have roles for all columns
        if check_target_and_folds:
            assert sds.target_column in sds.data.columns
            assert isinstance(sds.data.schema[sds.target_column].dataType, NumericType)

        assert SparkDataset.ID_COLUMN in sds.data.columns
        assert set(sds.features).issubset(sds.roles.keys())
        assert all(f in sds.data.columns for f in sds.features)

        if check_target_and_folds:
            assert not sds.folds_column or sds.folds_column in sds.data.columns

    # path = "../../examples/data/sampled_app_train.csv"
    # task_type = "binary"

    path = config['path']
    task_type = config['task_type']
    roles = config['roles']
    dtype = config['dtype'] if 'dtype' in config else None

    df = spark.read.csv(path, header=True, escape="\"")
    persistence_manager = PlainCachePersistenceManager()
    sreader = SparkToSparkReader(task=SparkTask(task_type), cv=cv, advanced_roles=agr)

    sdataset = sreader.fit_read(df, roles=roles, persistence_manager=persistence_manager)
    checks(sdataset)

    sdataset = sreader.read(df)
    assert sdataset.target_column is None
    assert sdataset.folds_column is None

    sdataset = sreader.read(df, add_array_attrs=True)
    checks(sdataset)

    # comparing with Pandas
    pdf = pd.read_csv(path, dtype=dtype)
    preader = PandasToPandasReader(task=Task(task_type), cv=cv, advanced_roles=agr)
    pdataset = preader.fit_read(pdf, roles=roles)

    assert set(sdataset.features) == set(pdataset.features)
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
    assert len(not_equal_encoding_types) == 0, f"Encoding types are different: {not_equal_encoding_types}"


# noinspection PyShadowingNames
@pytest.mark.parametrize("config,cv", [(ds, 5) for ds in get_test_datasets(dataset='used_cars_dataset_no_cols_limit')])
def test_spark_reader_advanced_guess_roles(spark: SparkSession, config: Dict[str, Any], cv: int):
    task_type = config['task_type']

    read_csv_args = {'dtype': config['dtype']} if 'dtype' in config else dict()
    train_pdf = pd.read_csv(config['train_path'], **read_csv_args)
    preader = PandasToPandasReader(task=Task(task_type), cv=cv)
    pdataset = preader.fit_read(train_pdf, roles=config["roles"])

    train_df = spark.read.csv(config['train_path'], header=True, escape="\"")
    persistence_manager = PlainCachePersistenceManager()
    sreader = SparkToSparkReader(task=SparkTask(task_type), cv=cv)
    sdataset = sreader.fit_read(train_df, roles=config["roles"], persistence_manager=persistence_manager)

    assert set(sdataset.features) == set(pdataset.features)
    sdiff = set(sdataset.features).symmetric_difference(pdataset.features)
    assert len(sdiff) == 0, f"Features sets are different: {sdiff}"
