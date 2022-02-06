from typing import cast, Dict, Any, List

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


def datasets(setting: str = "all") -> List[Dict[str, Any]]:
    used_cars_dataset = {
        "path": "examples/data/small_used_cars_data.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["dealer_zip", "description", "listed_date",
                     "year", 'Unnamed: 0', '_c0',
                     'sp_id', 'sp_name', 'trimId',
                     'trim_name', 'major_options', 'main_picture_url',
                     'interior_color', 'exterior_color'],
            "numeric": ['latitude', 'longitude', 'mileage']
        },
        "dtype": {
            'fleet': 'str', 'frame_damaged': 'str',
            'has_accidents': 'str', 'isCab': 'str',
            'is_cpo': 'str', 'is_new': 'str',
            'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
        }
    }

    lama_test_dataset = {
        "path": "./examples/data/sampled_app_train.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "TARGET",
        "roles": {"target": "TARGET", "drop": ["SK_ID_CURR"]},
    }

    # https://www.openml.org/d/734
    ailerons_dataset = {
        "path": "/opt/ailerons.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "binaryClass",
        "roles": {"target": "binaryClass"},
    }

    # https://www.openml.org/d/4534
    phishing_websites_dataset = {
        "path": "/opt/PhishingWebsites.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "Result",
        "roles": {"target": "Result"},
    }

    # https://www.openml.org/d/981
    kdd_internet_usage = {
        "path": "/opt/kdd_internet_usage.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "Who_Pays_for_Access_Work",
        "roles": {"target": "Who_Pays_for_Access_Work"},
    }

    # https://www.openml.org/d/42821
    nasa_dataset = {
        "path": "/opt/nasa_phm2008.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "class",
        "roles": {"target": "class"},
    }

    # https://www.openml.org/d/4549
    buzz_dataset = {
        "path": "/opt/Buzzinsocialmedia_Twitter_25k.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "Annotation",
        "roles": {"target": "Annotation"},
    }

    # https://www.openml.org/d/372
    internet_usage = {
        "path": "/opt/internet_usage.csv",
        "task_type": "multiclass",
        "metric_name": "ova",
        "target_col": "Actual_Time",
        "roles": {"target": "Actual_Time"},
    }

    # https://www.openml.org/d/4538
    gesture_segmentation = {
        "path": "/opt/gesture_segmentation.csv",
        "task_type": "multiclass",
        "metric_name": "ova",
        "target_col": "Phase",
        "roles": {"target": "Phase"},
    }

    # https://www.openml.org/d/382
    ipums_97 = {
        "path": "/opt/ipums_97.csv",
        "task_type": "multiclass",
        "metric_name": "ova",
        "target_col": "movedin",
        "roles": {"target": "movedin"},
    }

    if setting == "fast":
        return [used_cars_dataset]
    elif setting == "multiclass":
        return [internet_usage, gesture_segmentation, ipums_97]
    elif setting == "all":
        return [
            used_cars_dataset, lama_test_dataset, ailerons_dataset,
            phishing_websites_dataset, kdd_internet_usage, nasa_dataset,
            buzz_dataset, internet_usage, gesture_segmentation, ipums_97
        ]
    else:
        raise ValueError(f"Unsupported setting {setting}")


@pytest.mark.parametrize("config,cv", [(ds, 5) for ds in datasets(setting="fast")])
def test_spark_reader(spark: SparkSession, config: Dict[str, Any], cv: int):
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

    path = config['path']
    task_type = config['task_type']
    roles = config['roles']
    dtype = config['dtype'] if 'dtype' in config else None

    df = spark.read.csv(path, header=True, escape="\"")
    sreader = SparkToSparkReader(task=SparkTask(task_type), cv=cv)

    sdataset = sreader.fit_read(df, roles=roles)
    checks(sdataset)

    sdataset = sreader.read(df)
    checks(sdataset, check_target_and_folds=False)

    # comparing with Pandas
    pdf = pd.read_csv(path, dtype=dtype)
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
    assert len(not_equal_encoding_types) == 0, f"Encoding types are different: {not_equal_encoding_types}"
