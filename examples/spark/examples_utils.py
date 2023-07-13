import inspect
import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, Any, Dict

from pyspark import SparkContext
from pyspark.sql import SparkSession

from sparklightautoml.dataset import persistence
from sparklightautoml.utils import SparkDataFrame, get_current_session

BUCKET_NUMS = 6
PERSISTENCE_MANAGER_ENV_VAR = "PERSISTENCE_MANAGER"
BASE_DATASETS_PATH = "file:///opt/spark_data/"


@dataclass(frozen=True)
class Dataset:
    path: str
    task_type: str
    roles: Dict[str, Any]
    dtype: Dict[str, str] = field(default_factory=dict)
    file_format: str = 'csv'
    file_format_options: Dict[str, Any] = field(default_factory=lambda: {"header": True, "escape": "\""})

    def load(self) -> SparkDataFrame:
        spark = get_current_session()
        return spark.read.format(self.file_format).options(**self.file_format_options).load(self.path)


def ds_path(rel_path: str) -> str:
    return os.path.join(BASE_DATASETS_PATH, rel_path)


used_cars_params = {
    "task_type": "reg",
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

DATASETS = {
    "used_cars_dataset": Dataset(
        path=ds_path("small_used_cars_data.csv"),
        **used_cars_params
    ),
    "used_cars_dataset_1x": Dataset(
        path=ds_path("derivative_datasets/1x_dataset.csv"),
        **used_cars_params
    ),
    "used_cars_dataset_4x": Dataset(
        path=ds_path("derivative_datasets/4x_dataset.csv"),
        **used_cars_params
    ),
    "lama_test_dataset": Dataset(
        path=ds_path("sampled_app_train.csv"),
        task_type="binary",
        roles={"target": "TARGET", "drop": ["SK_ID_CURR"]}
    ),
    # https://www.openml.org/d/4549
    "buzz_dataset": Dataset(
        path=ds_path("Buzzinsocialmedia_Twitter_25k.csv"),
        task_type="binary",
        roles={"target": "TARGET", "drop": ["SK_ID_CURR"]}
    ),
    # https://www.openml.org/d/734
    "ailerons_dataset": Dataset(
        path=ds_path("ailerons.csv"),
        task_type="binary",
        roles={"target": "binaryClass"}
    ),
    # https://www.openml.org/d/382
    "ipums_97": Dataset(
        path=ds_path("ipums_97.csv"),
        task_type="multiclass",
        roles={"target": "movedin"}
    ),

    "company_bankruptcy_dataset": Dataset(
        path=ds_path("company_bankruptcy_prediction_data.csv"),
        task_type="binary",
        roles={"target": "Bankrupt?"}
    )
}


def get_dataset(name: str) -> Dataset:
    assert name in DATASETS, f"Unknown dataset: {name}. Known datasets: {list(DATASETS.keys())}"
    return DATASETS[name]


def prepare_test_and_train(
        dataset: Dataset,
        seed: int,
        test_size: float = 0.2
) -> Tuple[SparkDataFrame, SparkDataFrame]:
    assert 0 <= test_size <= 1

    spark = get_current_session()

    execs = int(spark.conf.get('spark.executor.instances', '1'))
    cores = int(spark.conf.get('spark.executor.cores', '8'))

    data = dataset.load()

    data = data.repartition(execs * cores).cache()
    data.write.mode('overwrite').format('noop').save()

    train_data, test_data = data.randomSplit([1 - test_size, test_size], seed)
    train_data = train_data.cache()
    test_data = test_data.cache()
    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    data.unpersist()

    return train_data, test_data


def get_spark_session(partitions_num: Optional[int] = None):
    partitions_num = partitions_num if partitions_num else BUCKET_NUMS

    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        spark_sess = SparkSession.builder.getOrCreate()
    else:

        # Be aware, this an alternative way to supply SLAMA with its jars using maven repository
        # Example requesting both synapseml and SLAMA jar from Maven Central
        # .config("spark.jars.packages",
        #         "com.microsoft.azure:synapseml_2.12:0.9.5,io.github.fonhorst:spark-lightautoml_2.12:0.1.1")

        spark_sess = (
            SparkSession
            .builder
            .master(f"local[{partitions_num}]")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
            .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.1.jar")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
            .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryoserializer.buffer.max", "512m")
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
            .config("spark.cleaner.referenceTracking", "true")
            .config("spark.cleaner.periodicGC.interval", "1min")
            .config("spark.sql.shuffle.partitions", f"{partitions_num}")
            .config("spark.default.parallelism", f"{partitions_num}")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.autoBroadcastJoinThreshold", "-1")
            .getOrCreate()
        )

    spark_sess.sparkContext.setCheckpointDir("/tmp/spark_checkpoints")

    spark_sess.sparkContext.setLogLevel("WARN")

    return spark_sess


def get_persistence_manager(name: Optional[str] = None):
    arg_vals = {
        "bucketed_datasets_folder": "/tmp",
        "bucket_nums": BUCKET_NUMS
    }

    class_name = name or os.environ.get(PERSISTENCE_MANAGER_ENV_VAR, None) or "CompositeBucketedPersistenceManager"
    clazz = getattr(persistence, class_name)
    sig = inspect.signature(getattr(clazz, "__init__"))

    ctr_arg_vals = {
        name: arg_vals.get(name, None if p.default is p.empty else p.default)
        for name, p in sig.parameters.items() if name != 'self'
    }

    none_val_args = [name for name, val in ctr_arg_vals.items() if val is None]
    assert len(none_val_args) == 0, f"Cannot instantiate class {class_name}. " \
                                    f"Values for the following arguments have not been found: {none_val_args}"

    return clazz(**ctr_arg_vals)


class FSOps:
    """
        Set of typical fs operations independent of the fs implementation
        see docs at: https://hadoop.apache.org/docs/current/api/org/apache/hadoop/fs/FileSystem.html
    """
    @staticmethod
    def get_sc() -> SparkContext:
        spark = get_current_session()
        sc = spark.sparkContext
        return sc

    @staticmethod
    def get_default_fs() -> str:
        spark = get_current_session()
        hadoop_conf = spark._jsc.hadoopConfiguration()
        default_fs = hadoop_conf.get("fs.defaultFS")
        return default_fs

    @classmethod
    def get_fs(cls, path: str):
        sc = cls.get_sc()

        URI = sc._jvm.java.net.URI
        FileSystem = sc._jvm.org.apache.hadoop.fs.FileSystem
        Configuration = sc._jvm.org.apache.hadoop.conf.Configuration

        path_uri = URI(path)
        scheme = path_uri.getScheme()
        if scheme:
            authority = path_uri.getAuthority() or ''
            fs_uri = f'{scheme}:/{authority}'
        else:
            fs_uri = cls.get_default_fs()

        fs = FileSystem.get(URI(fs_uri), Configuration())

        return fs

    @classmethod
    def exists(cls, path: str) -> bool:
        sc = cls.get_sc()
        Path = sc._jvm.org.apache.hadoop.fs.Path
        fs = cls.get_fs(path)
        return fs.exists(Path(path))

    @classmethod
    def create_dir(cls, path: str):
        sc = cls.get_sc()
        Path = sc._jvm.org.apache.hadoop.fs.Path
        fs = cls.get_fs(path)
        fs.mkdirs(Path(path))

    @classmethod
    def delete_dir(cls, path: str) -> bool:
        sc = cls.get_sc()
        Path = sc._jvm.org.apache.hadoop.fs.Path
        fs = cls.get_fs(path)
        return fs.delete(Path('/tmp/just_a_test'))


def check_columns(original_df: SparkDataFrame, predicts_df: SparkDataFrame):
    absent_columns = set(original_df.columns).difference(predicts_df.columns)
    assert len(absent_columns) == 0, \
        f"Some columns of the original dataframe is absent from the processed dataset: {absent_columns}"
