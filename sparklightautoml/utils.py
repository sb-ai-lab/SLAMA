import logging
import os
import socket
import time
import warnings

from contextlib import contextmanager
from datetime import datetime
from logging import Logger
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import cast

import pyspark

from py4j.java_gateway import java_import
from pyspark import RDD
from pyspark.ml import Estimator
from pyspark.ml import Transformer
from pyspark.ml.common import inherit_doc
from pyspark.ml.param import Param
from pyspark.ml.param import Params
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.param.shared import HasOutputCols
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.pipeline import PipelineSharedReadWrite
from pyspark.ml.util import DefaultParamsReadable
from pyspark.ml.util import DefaultParamsReader
from pyspark.ml.util import DefaultParamsWritable
from pyspark.ml.util import DefaultParamsWriter
from pyspark.ml.util import MLReader
from pyspark.ml.util import MLWriter
from pyspark.sql import SparkSession


VERBOSE_LOGGING_FORMAT = "%(asctime)s %(threadName)s %(levelname)s %(module)s %(filename)s:%(lineno)d %(message)s"

logger = logging.getLogger(__name__)


@contextmanager
def spark_session(
    session_args: Optional[dict] = None, master: str = "local[]", wait_secs_after_the_end: Optional[int] = None
) -> SparkSession:
    """
    Args:
        session_args: additional arguments to be add to SparkSession using .config() method
        master: address of the master
            to run locally - "local[1]"

            to run on spark cluster - "spark://node4.bdcl:7077"
            (Optionally set the driver host to a correct hostname .config("spark.driver.host", "node4.bdcl"))

        wait_secs_after_the_end: amount of seconds to wait before stoping SparkSession and thus web UI.

    Returns:
        SparkSession to be used and that is stopped upon exiting this context manager
    """

    if not session_args:
        spark_sess_builder = (
            SparkSession.builder.appName("SPARK-LAMA-app")
            .master(master)
            .config("spark.jars", "jars/spark-lightautoml_2.12-0.1.jar")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.kryoserializer.buffer.max", "512m")
            .config("spark.driver.cores", "4")
            .config("spark.driver.memory", "16g")
            .config("spark.cores.max", "16")
            .config("spark.executor.instances", "4")
            .config("spark.executor.memory", "16g")
            .config("spark.executor.cores", "4")
            .config("spark.memory.fraction", "0.6")
            .config("spark.memory.storageFraction", "0.5")
            .config("spark.sql.autoBroadcastJoinThreshold", "100MB")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        )
    else:
        spark_sess_builder = SparkSession.builder.appName("SPARK-LAMA-app")
        for arg, value in session_args.items():
            spark_sess_builder = spark_sess_builder.config(arg, value)

        if "spark.master" in session_args and not session_args["spark.master"].startswith("local["):
            local_ip_address = socket.gethostbyname(socket.gethostname())
            logger.info(f"Using IP address for spark driver: {local_ip_address}")
            spark_sess_builder = spark_sess_builder.config("spark.driver.host", local_ip_address)

    spark_sess = spark_sess_builder.getOrCreate()

    logger.info(f"Spark WebUI url: {spark_sess.sparkContext.uiWebUrl}")

    try:
        yield spark_sess
    finally:
        logger.info(
            f"The session is ended. Sleeping {wait_secs_after_the_end if wait_secs_after_the_end else 0} "
            f"secs until stop the spark session."
        )
        if wait_secs_after_the_end:
            time.sleep(wait_secs_after_the_end)
        spark_sess.stop()


@contextmanager
def log_exec_time(name: Optional[str] = None, write_log=True):
    # Add file handler for INFO
    if write_log:
        file_handler_info = logging.FileHandler(f"/tmp/{name}_log.log.log", mode="a")
        file_handler_info.setFormatter(logging.Formatter("%(message)s"))
        file_handler_info.setLevel(logging.INFO)
        logger.addHandler(file_handler_info)

    start = datetime.now()

    yield

    end = datetime.now()
    duration = (end - start).total_seconds()

    msg = f"Exec time of {name}: {duration}" if name else f"Exec time: {duration}"
    logger.warning(msg)


# log_exec_time() class to return elapsed time value
class log_exec_timer:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._start = None
        self._duration = None

    def __enter__(self):
        self._start = datetime.now()
        return self

    def __exit__(self, typ, value, traceback):
        self._duration = (datetime.now() - self._start).total_seconds()
        msg = f"Exec time of {self.name}: {self._duration}" if self.name else f"Exec time: {self._duration}"
        logger.info(msg)

    @property
    def duration(self):
        return self._duration


SparkDataFrame = pyspark.sql.DataFrame


def get_current_session() -> SparkSession:
    return SparkSession.builder.getOrCreate()


def get_cached_df_through_rdd(df: SparkDataFrame, name: Optional[str] = None) -> Tuple[SparkDataFrame, RDD]:
    rdd = df.rdd
    cached_rdd = rdd.setName(name).cache() if name else rdd.cache()
    cached_df = df.sql_ctx.createDataFrame(cached_rdd, df.schema)
    return cached_df, cached_rdd


def logging_config(level: int = logging.INFO, log_filename: str = "/var/log/lama.log") -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "verbose": {"format": VERBOSE_LOGGING_FORMAT},
            "simple": {"format": "%(asctime)s %(levelname)s %(message)s"},
        },
        "handlers": {
            "console": {"level": "DEBUG", "class": "logging.StreamHandler", "formatter": "verbose"},
            "file": {
                "level": "DEBUG",
                "class": "logging.FileHandler",
                "formatter": "verbose",
                "filename": log_filename,
            },
        },
        "loggers": {
            "lightautoml": {
                "handlers": ["console", "file"],
                "propagate": True,
                "level": level,
            },
            "sparklightautoml": {
                "handlers": ["console", "file"],
                "level": level,
                "propagate": False,
            },
            "lightautoml.ml_algo": {
                "handlers": ["console", "file"],
                "level": level,
                "propagate": False,
            },
        },
    }


def cache(df: SparkDataFrame) -> SparkDataFrame:
    if not df.is_cached:
        df = df.cache()
    return df


def warn_if_not_cached(df: SparkDataFrame):
    if not df.is_cached:
        warnings.warn(
            "Attempting to calculate shape on not cached dataframe. " "It may take too much time.", RuntimeWarning
        )


class ColumnsSelectorTransformer(
    Transformer, HasInputCols, HasOutputCols, DefaultParamsWritable, DefaultParamsReadable
):
    """
    Makes selection input columns from input dataframe.
    """

    optionalCols = Param(
        Params._dummy(), "optionalCols", "optional column names.", typeConverter=TypeConverters.toListString
    )

    transformOnlyFirstTime = Param(
        Params._dummy(),
        "transformOnlyFirstTime",
        "whatever to transform only once or each time",
        typeConverter=TypeConverters.toBoolean,
    )

    _alreadyTransformed = Param(
        Params._dummy(),
        "_alreadyTransformed",
        "is it first time to transform or not",
        typeConverter=TypeConverters.toBoolean,
    )

    def __init__(
        self,
        name: Optional[str] = None,
        input_cols: Optional[List[str]] = None,
        optional_cols: Optional[List[str]] = None,
        transform_only_first_time: bool = False,
    ):
        super().__init__()
        input_cols = input_cols if input_cols else []
        optional_cols = optional_cols if optional_cols else []
        assert (
            len(set(input_cols).intersection(set(optional_cols))) == 0
        ), "Input columns and optional columns cannot intersect"

        self._name = name
        self.set(self.inputCols, input_cols)
        self.set(self.optionalCols, optional_cols)
        self.set(self.outputCols, input_cols)
        self.set(self.transformOnlyFirstTime, transform_only_first_time)
        self.set(self._alreadyTransformed, False)

    def get_optional_cols(self) -> List[str]:
        return self.getOrDefault(self.optionalCols)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        logger.debug(f"In {type(self)}. Name: {self._name}. Columns: {sorted(dataset.columns)}")

        if not self.getOrDefault(self.transformOnlyFirstTime) or not self.getOrDefault(self._alreadyTransformed):
            ds_cols = set(dataset.columns)
            present_opt_cols = [c for c in self.get_optional_cols() if c in ds_cols]
            input_cols = self._treat_columns_pattern(dataset, self.getInputCols())
            opt_input_cols = self._treat_columns_pattern(dataset, present_opt_cols)
            dataset = dataset.select([*input_cols, *opt_input_cols])
            self.set(self._alreadyTransformed, True)
            self.set(self.outputCols, input_cols)

        logger.debug(f"Out {type(self)}. Name: {self._name}. Columns: {sorted(dataset.columns)}")
        return dataset

    @staticmethod
    def _treat_columns_pattern(df: SparkDataFrame, cols: List[str]) -> List[str]:
        def treat(col: str):
            if col.endswith("*"):
                pattern = col[:-1]
                return [c for c in df.columns if c.startswith(pattern)]
            return [col]

        cols = [cc for c in cols for cc in treat(c)]
        return cols


class NoOpTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self._name = name

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        logger.debug(f"In {type(self)}. Name: {self._name}. Columns: {sorted(dataset.columns)}")
        return dataset


@inherit_doc
class WrappingSelectingPipelineModelWriter(MLWriter):
    """
    (Private) Specialization of :py:class:`MLWriter` for :py:class:`PipelineModel` types
    """

    def __init__(self, instance: "WrappingSelectingPipelineModel"):
        super(WrappingSelectingPipelineModelWriter, self).__init__()
        self.instance = instance

    def saveImpl(self, path: str):
        stages = self.instance.stages
        PipelineSharedReadWrite.validateStages(stages)

        stageUids = [stage.uid for stage in stages]
        jsonParams = {
            "stageUids": stageUids,
            "language": "Python",
            "instance_params": {param.name: value for param, value in self.instance.extractParamMap().items()},
        }
        DefaultParamsWriter.saveMetadata(self.instance, path, self.sc, paramMap=jsonParams)
        stagesDir = os.path.join(path, "stages")
        for index, stage in enumerate(stages):
            stage.write().save(PipelineSharedReadWrite.getStagePath(stage.uid, index, len(stages), stagesDir))


@inherit_doc
class WrappingSelectionPipelineModelReader(MLReader):
    """
    (Private) Specialization of :py:class:`MLReader` for :py:class:`PipelineModel` types
    """

    def __init__(self, cls):
        super(WrappingSelectionPipelineModelReader, self).__init__()
        self.cls = cls

    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)

        uid, stages = PipelineSharedReadWrite.load(metadata, self.sc, path)
        instance_params_map = cast(Dict, metadata["paramMap"]["instance_params"])

        wspm = WrappingSelectingPipelineModel(stages=stages)
        wspm._resetUid(uid)
        for param_name, value in instance_params_map.items():
            param = wspm.getParam(param_name)
            wspm.set(param, value)

        return wspm


class WrappingSelectingPipelineModel(PipelineModel, HasInputCols):
    name = Param(Params._dummy(), "name", "name.", typeConverter=TypeConverters.toString)

    optionalCols = Param(
        Params._dummy(), "optionalCols", "optional column names.", typeConverter=TypeConverters.toListString
    )

    def __init__(
        self,
        stages: List[Transformer],
        input_columns: Optional[List[str]] = None,
        optional_columns: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(stages)
        self.set(self.inputCols, input_columns or [])
        self.set(self.optionalCols, optional_columns or [])
        self.set(self.name, name or "")

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        cstr = ColumnsSelectorTransformer(
            name=f"{type(self).__name__}({self.getOrDefault(self.name)})",
            input_cols=list({*dataset.columns, *self.getInputCols()}),
            optional_cols=self.getOrDefault(self.optionalCols),
        )
        ds = super()._transform(dataset)
        return cstr.transform(ds)

    def write(self):
        """Returns an MLWriter instance for this ML instance."""
        return WrappingSelectingPipelineModelWriter(self)

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return WrappingSelectionPipelineModelReader(cls)


class Cacher(Estimator):
    _cacher_dict: Dict[str, SparkDataFrame] = dict()

    @classmethod
    def get_dataset_by_key(cls, key: str) -> Optional[SparkDataFrame]:
        return cls._cacher_dict.get(key, None)

    @classmethod
    def release_cache_by_key(cls, key: str):
        df = cls._cacher_dict.pop(key, None)
        if df is not None:
            df.unpersist()
            del df

    @property
    def dataset(self) -> SparkDataFrame:
        """Returns chached dataframe"""
        return self._cacher_dict[self._key]

    def __init__(self, key: str):
        super().__init__()
        self._key = key
        self._dataset: Optional[SparkDataFrame] = None

    def _fit(self, dataset):
        logger.info(f"Cacher {self._key} (RDD Id: {dataset.rdd.id()}). Starting to materialize data.")

        # using local checkpoints
        # ds = dataset.localCheckpoint(eager=True)

        # # using plain caching
        # ds = get_current_session().createDataFrame(dataset.rdd, schema=dataset.schema).cache()
        ds = dataset.cache()
        ds.write.mode("overwrite").format("noop").save()

        logger.info(
            f"Cacher {self._key} (RDD Id: {ds.rdd.id()}, Column nums: {len(ds.columns)}). "
            f"Finished data materialization."
        )

        previous_ds = self._cacher_dict.get(self._key, None)
        if previous_ds is not None and ds != previous_ds:
            logger.info(f"Removing cache for key: {self._key} (RDD Id: {previous_ds.rdd.id()}).")
            previous_ds.unpersist()
            del previous_ds

        self._cacher_dict[self._key] = ds

        return NoOpTransformer(name=f"cacher_{self._key}")


class EmptyCacher(Cacher):
    def __init__(self, key: str):
        super().__init__(key)
        self._dataset: Optional[SparkDataFrame] = None

    @property
    def dataset(self) -> SparkDataFrame:
        return self._dataset

    def _fit(self, dataset):
        self._dataset = dataset
        return NoOpTransformer(name=f"empty_cacher_{self._key}")


def log_exception(logger: Logger):
    def wrap(func):
        def wrapped_f(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as ex:
                logger.error("Error wrapper caught error", exc_info=True)
                raise ex
            return result

        return wrapped_f

    return wrap


@contextmanager
def JobGroup(group_id: str, description: str, spark: SparkSession):
    sc = spark.sparkContext
    sc.setJobGroup(group_id, description)
    yield
    sc._jsc.clearJobGroup()


# noinspection PyProtectedMember,PyUnresolvedReferences
def create_directory(path: str, spark: SparkSession, exists_ok: bool = False):
    java_import(spark._jvm, "org.apache.hadoop.fs.Path")
    java_import(spark._jvm, "java.net.URI")
    java_import(spark._jvm, "org.apache.hadoop.fs.FileSystem")

    juri = spark._jvm.Path(path).toUri()
    jpath = spark._jvm.Path(juri.getPath())
    jscheme = spark._jvm.URI(f"{juri.getScheme()}://{juri.getAuthority() or ''}/") if juri.getScheme() else None
    fs = (
        spark._jvm.FileSystem.get(jscheme, spark._jsc.hadoopConfiguration())
        if jscheme
        else spark._jvm.FileSystem.get(spark._jsc.hadoopConfiguration())
    )

    if not fs.exists(jpath):
        fs.mkdirs(jpath)
    elif not exists_ok:
        raise FileExistsError(f"The path already exists: {path}")
