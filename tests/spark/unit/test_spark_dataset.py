import json
import os
import shutil
from datetime import datetime
from typing import Optional

import numpy as np
from lightautoml.dataset.roles import NumericRole, CategoryRole, DatetimeRole, DropRole, TextRole, DateRole, IdRole, \
    TargetRole, GroupRole, WeightsRole, FoldsRole, PathRole, TreatmentRole
from lightautoml.tasks import Task
from pandas.testing import assert_frame_equal
from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset, SparkDatasetMetadataJsonEncoder, SparkDatasetMetadataJsonDecoder
from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.tasks.base import SparkTask
from . import spark as spark_sess

spark = spark_sess


def compare_tasks(task_a: Optional[Task], task_b: Optional[Task]):
    assert (task_a and task_b) or (not task_a and not task_b)
    assert task_a.name == task_b.name
    assert task_a.metric_name == task_b.metric_name
    assert task_a.greater_is_better == task_b.greater_is_better


def compare_dfs(dataset_a: SparkDataset, dataset_b: SparkDataset):
    assert dataset_a.data.schema == dataset_b.data.schema

    # checking data
    df_a = dataset_a.data.orderBy(SparkDataset.ID_COLUMN).toPandas()
    df_b = dataset_b.data.orderBy(SparkDataset.ID_COLUMN).toPandas()
    assert_frame_equal(df_a, df_b)


def test_column_roles_json_encoder_and_decoder():
    roles = {
        "num_role": NumericRole(),
        "num_vect_role": NumericVectorOrArrayRole(size=20, element_col_name_template="some_template"),
        "cat_role": CategoryRole(),
        "id_role": IdRole(),
        "dt_role": DatetimeRole(origin=datetime.now()),
        "d_role": DateRole(),
        "text_role": TextRole(),
        "target_role": TargetRole(),
        "group_role": GroupRole(),
        "drop_role": DropRole(),
        "weights_role": WeightsRole(),
        "folds_role": FoldsRole(),
        "path_role": PathRole(),
        "treatment_role": TreatmentRole()
    }

    js_roles = json.dumps(roles, cls=SparkDatasetMetadataJsonEncoder)
    deser_roles = json.loads(js_roles, cls=SparkDatasetMetadataJsonDecoder)

    assert deser_roles == roles


def test_spark_task_json_encoder_decoder():
    stask = SparkTask("binary")

    js_stask = json.dumps(stask, cls=SparkDatasetMetadataJsonEncoder)
    deser_stask = json.loads(js_stask, cls=SparkDatasetMetadataJsonDecoder)

    stask_internals = [stask.name, stask.loss_name, stask.metric_name, stask.greater_is_better]
    deser_stask_internals = [deser_stask.name, deser_stask.loss_name,
                             deser_stask.metric_name, deser_stask.greater_is_better]

    assert deser_stask_internals == stask_internals


def test_spark_dataset_save_load(spark: SparkSession):
    path = "/tmp/test_slama_ds.dataset"
    partitions_num = 37

    # cleanup
    if os.path.exists(path):
        shutil.rmtree(path)

    # creating test data
    df = spark.createDataFrame([{
        SparkDataset.ID_COLUMN: i,
        "a": i + 1,
        "b": i * 10 + 1,
        "this_is_target": 0,
        "this_is_fold": 0,
        "scaler__fillnamed__fillinf__logodds__oof__inter__(CODE_GENDER__EMERGENCYSTATE_MODE)": 12.0
    } for i in range(10)])

    ds = SparkDataset(
        data=df,
        task=SparkTask("reg"),
        target="this_is_target",
        folds="this_is_fold",
        roles={
            "a": NumericRole(dtype=np.int32),
            "b": NumericRole(dtype=np.int32),
            "scaler__fillnamed__fillinf__logodds__oof__inter__(CODE_GENDER__EMERGENCYSTATE_MODE)": NumericRole()
        }
    )

    ds.save(path=path)
    loaded_ds = SparkDataset.load(path=path, partitions_num=partitions_num)

    # checking metadata
    assert loaded_ds.uid
    assert loaded_ds.uid != ds.uid
    assert loaded_ds.name == ds.name
    assert loaded_ds.target_column == ds.target_column
    assert loaded_ds.folds_column == ds.folds_column
    assert loaded_ds.service_columns == ds.service_columns
    assert loaded_ds.features == ds.features
    assert loaded_ds.roles == ds.roles
    assert loaded_ds.data.rdd.getNumPartitions() == partitions_num
    compare_tasks(loaded_ds.task, ds.task)
    compare_dfs(loaded_ds, ds)

    # cleanup
    if os.path.exists(path):
        shutil.rmtree(path)
