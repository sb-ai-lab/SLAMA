from typing import Dict, Any

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType

from lightautoml.dataset.roles import CategoryRole
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.tasks.base import SparkTask as SparkTask
from lightautoml.tasks import Task
from . import spark as spark_sess
from ..dataset_utils import get_test_datasets

spark = spark_sess


@pytest.mark.parametrize("config,cv", [(ds, 5) for ds in get_test_datasets(setting="all-tasks")])
def test_spark_reader(spark: SparkSession, config: Dict[str, Any], cv: int):
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
    sreader = SparkToSparkReader(task=SparkTask(task_type), cv=cv, advanced_roles=False)

    sdataset = sreader.fit_read(df, roles=roles)
    checks(sdataset)

    sdataset = sreader.read(df, add_array_attrs=False)
    assert sdataset.target_column is None
    assert sdataset.folds_column is None
    assert roles['target'] not in sdataset.data.columns

    sdataset = sreader.read(df, add_array_attrs=True)
    checks(sdataset, check_target_and_folds=True)

    # comparing with Pandas
    pdf = pd.read_csv(path, dtype=dtype)
    preader = PandasToPandasReader(task=Task(task_type), cv=cv)
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
