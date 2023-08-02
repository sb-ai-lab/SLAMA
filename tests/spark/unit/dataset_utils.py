import os
import pickle
import shutil

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pyspark.sql.functions as sf

from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import PersistenceManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask


DUMP_METADATA_NAME = "metadata.pickle"
DUMP_DATA_NAME = "data.parquet"


def dump_data(path: str, ds: SparkDataset, **meta_kwargs):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)

    metadata_file = os.path.join(path, DUMP_METADATA_NAME)
    data_file = os.path.join(path, DUMP_DATA_NAME)

    metadata = {
        "roles": ds.roles,
        "target": ds.target_column,
        "folds": ds.folds_column,
        "task_name": ds.task.name if ds.task else None,
    }
    metadata.update(meta_kwargs)

    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

    cols_to_rename = [sf.col(c).alias(c.replace("(", "[").replace(")", "]")) for c in ds.data.columns]

    ds.data.select(*cols_to_rename).write.parquet(data_file)


def load_dump_if_exist(
    spark: SparkSession, persistence_manager: PersistenceManager, path: Optional[str] = None
) -> Optional[Tuple[SparkDataset, Dict]]:
    if path is None:
        return None

    if not os.path.exists(path):
        return None

    metadata_file = os.path.join(path, DUMP_METADATA_NAME)
    data_file = os.path.join(path, DUMP_DATA_NAME)

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    df = spark.read.parquet(data_file)

    cols_to_rename = [sf.col(c).alias(c.replace("[", "(").replace("]", ")")) for c in df.columns]

    df = df.select(*cols_to_rename).repartition(16).cache()
    df.write.mode("overwrite").format("noop").save()

    ds = SparkDataset(
        data=df,
        roles=metadata["roles"],
        persistence_manager=persistence_manager,
        task=SparkTask(metadata["task_name"]),
        target=metadata["target"],
        folds=metadata["folds"],
    )

    return ds, metadata


all_datastes = {
    "used_cars_dataset": {
        "path": "examples/data/small_used_cars_data.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": [
                "dealer_zip",
                "description",
                "listed_date",
                "year",
                "Unnamed: 0",
                "_c0",
                "sp_id",
                "sp_name",
                "trimId",
                "trim_name",
                "major_options",
                "main_picture_url",
                "interior_color",
                "exterior_color",
            ],
            # "numeric": ['latitude', 'longitude', 'mileage']
            "numeric": ["longitude", "mileage"],
        },
        "dtype": {
            "fleet": "str",
            "frame_damaged": "str",
            "has_accidents": "str",
            "isCab": "str",
            "is_cpo": "str",
            "is_new": "str",
            "is_oemcpo": "str",
            "salvage": "str",
            "theft_title": "str",
            "franchise_dealer": "str",
        },
    },

    "used_cars_dataset_no_cols_limit": {
        "path": "examples/data/small_used_cars_data.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "price",
        "roles": {
            "target": "price",
            "drop": ["Unnamed: 0", "_c0"],
        },
        "dtype": {
            "fleet": "str",
            "frame_damaged": "str",
            "has_accidents": "str",
            "isCab": "str",
            "is_cpo": "str",
            "is_new": "str",
            "is_oemcpo": "str",
            "salvage": "str",
            "theft_title": "str",
            "franchise_dealer": "str",
        },
    },

    "lama_test_dataset": {
        "path": "examples/data/sampled_app_train.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "TARGET",
        "roles": {"target": "TARGET", "drop": ["SK_ID_CURR"]},
    },

    # https://www.openml.org/d/734
    "ailerons_dataset": {
        "path": "examples/data/ailerons.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "binaryClass",
        "roles": {"target": "binaryClass"},
    },

    # https://www.openml.org/d/4534
    "phishing_websites_dataset": {
        "path": "examples/data/PhishingWebsites.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "Result",
        "roles": {"target": "Result"},
    },

    # https://www.openml.org/d/981
    "kdd_internet_usage": {
        "path": "examples/data/kdd_internet_usage.csv",
        "task_type": "binary",
        "metric_name": "areaUnderROC",
        "target_col": "Who_Pays_for_Access_Work",
        "roles": {"target": "Who_Pays_for_Access_Work"},
    },

    # https://www.openml.org/d/42821
    "nasa_dataset": {
        "path": "examples/data/nasa_phm2008.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "class",
        "roles": {"target": "class"},
    },

    # https://www.openml.org/d/4549
    "buzz_dataset": {
        "path": "examples/data/Buzzinsocialmedia_Twitter_25k.csv",
        "task_type": "reg",
        "metric_name": "mse",
        "target_col": "Annotation",
        "roles": {"target": "Annotation"},
    },

    # https://www.openml.org/d/372
    "internet_usage": {
        "path": "examples/data/internet_usage.csv",
        "task_type": "multiclass",
        "metric_name": "crossentropy",
        "target_col": "Actual_Time",
        "roles": {"target": "Actual_Time"},
    },

    # https://www.openml.org/d/4538
    "gesture_segmentation": {
        "path": "examples/data/gesture_segmentation.csv",
        "task_type": "multiclass",
        "metric_name": "crossentropy",
        "target_col": "Phase",
        "roles": {"target": "Phase"},
    },

    # https://www.openml.org/d/382
    "ipums_97": {
        "path": "examples/data/ipums_97.csv",
        "task_type": "multiclass",
        "metric_name": "crossentropy",
        "target_col": "movedin",
        "roles": {"target": "movedin"},
    }
}


def datasets() -> Dict[str, Any]:
    return all_datastes


def get_test_datasets(dataset: Optional[str] = None, setting: str = "all") -> List[Dict[str, Any]]:
    dss = datasets()

    if dataset is not None:
        return [dss[dataset]]

    if setting == "fast":
        return [dss["used_cars_dataset"]]
    elif setting == "multiclass":
        return [dss["gesture_segmentation"], dss["ipums_97"]]
    elif setting == "reg+binary":
        return [
            dss["used_cars_dataset"],
            dss["buzz_dataset"],
            dss["lama_test_dataset"],
            dss["ailerons_dataset"],
        ]
    elif setting == "binary":
        return [
            dss["lama_test_dataset"],
            dss["ailerons_dataset"],
        ]
    elif setting == "one_reg+one_binary":
        return [dss["used_cars_dataset"], dss["lama_test_dataset"]]
    elif setting == "all-tasks":
        return [
            dss["used_cars_dataset"],
            dss["buzz_dataset"],
            dss["lama_test_dataset"],
            dss["ailerons_dataset"],
            dss["gesture_segmentation"],
            dss["ipums_97"],
        ]
    elif setting == "all":
        # exccluding all heavy datasets
        return list(cfg for ds_name, cfg in dss.items() if not ds_name.startswith("used_cars_dataset_"))
    else:
        raise ValueError(f"Unsupported setting {setting}")
