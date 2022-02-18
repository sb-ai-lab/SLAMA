import logging
import os
import socket
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Tuple

from pyspark import RDD
from pyspark.sql import SparkSession

from lightautoml.spark.dataset.base import SparkDataFrame

VERBOSE_LOGGING_FORMAT = '%(asctime)s %(levelname)s %(module)s %(filename)s:%(lineno)d %(message)s'

logger = logging.getLogger(__name__)


@contextmanager
def spark_session(session_args: Optional[dict] = None, master: str = "local[]", wait_secs_after_the_end: Optional[int] = None) -> SparkSession:
    """
    Args:
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
            SparkSession
            .builder
            .appName("SPARK-LAMA-app")
            .master(master)
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .config("spark.sql.shuffle.partitions", "4")
            # .config("spark.driver.extraJavaOptions", "-Ddev.ludovic.netlib.blas.nativeLibPath=/usr/lib64/libopenblaso-r0.3.17.so")
            # .config("spark.executor.extraJavaOptions", "-Ddev.ludovic.netlib.blas.nativeLibPath=/usr/lib64/libopenblaso-r0.3.17.so")
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
        spark_sess_builder = (
            SparkSession
            .builder
            .appName("SPARK-LAMA-app")
        )
        for arg, value in session_args.items():
            spark_sess_builder = spark_sess_builder.config(arg, value)

        if 'spark.master' in session_args and not session_args['spark.master'].startswith('local['):
            local_ip_address = socket.gethostbyname(socket.gethostname())
            logger.info(f"Using IP address for spark driver: {local_ip_address}")
            spark_sess_builder = spark_sess_builder.config('spark.driver.host', local_ip_address)

    spark_sess = spark_sess_builder.getOrCreate()

    logger.info(f"Spark WebUI url: {spark_sess.sparkContext.uiWebUrl}")

    try:
        yield spark_sess
    finally:
        logger.info(f"The session is ended. Sleeping {wait_secs_after_the_end if wait_secs_after_the_end else 0} "
                    f"secs until stop the spark session.")
        if wait_secs_after_the_end:
            time.sleep(wait_secs_after_the_end)
        spark_sess.stop()


@contextmanager
def log_exec_time(name: Optional[str] = None):

    # Add file handler for INFO 
    file_handler_info = logging.FileHandler(f'/tmp/{name}_log.log.log', mode='a')
    file_handler_info.setFormatter(logging.Formatter('%(message)s'))
    file_handler_info.setLevel(logging.INFO)
    logger.addHandler(file_handler_info)

    start = datetime.now()

    yield 

    end = datetime.now()
    duration = (end - start).total_seconds()

    msg = f"Exec time of {name}: {duration}" if name else f"Exec time: {duration}"
    logger.info(msg)


# log_exec_time() class to return elapsed time value
class log_exec_timer:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        file_handler_info = logging.FileHandler(f'/tmp/{name}_log.log', mode='w')
        file_handler_info.setFormatter(logging.Formatter('%(message)s'))
        file_handler_info.setLevel(logging.INFO)
        logger.addHandler(file_handler_info)
        self._start = None
        self._duration = None

    def __enter__(self):
        self._start = datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        self._duration = (datetime.now() - self._start).total_seconds()
        msg = f"Exec time of {self.name}: {self._duration}" if self.name else f"Exec time: {self._duration}"
        logger.info(msg)

    @property
    def duration(self):
        return self._duration


def get_cached_df_through_rdd(df: SparkDataFrame, name: Optional[str] = None) -> Tuple[SparkDataFrame, RDD]:
    rdd = df.rdd
    cached_rdd = rdd.setName(name).cache() if name else rdd.cache()
    cached_df = df.sql_ctx.createDataFrame(cached_rdd, df.schema)
    return cached_df, cached_rdd


def logging_config(level: int = logging.INFO, log_filename: str = '/var/log/lama.log') -> dict:
    return {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'verbose': {
                'format': VERBOSE_LOGGING_FORMAT
            },
            'simple': {
                'format': '%(asctime)s %(levelname)s %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'verbose'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'formatter': 'verbose',
                'filename': log_filename
            }
        },
        'loggers': {
            'lightautoml': {
                'handlers': ['console', 'file'],
                'propagate': True,
                'level': level,
            },
            'lightautoml.spark': {
                'handlers': ['console', 'file'],
                'level': level,
                'propagate': False,
            },
            'lightautoml.ml_algo': {
                'handlers': ['console', 'file'],
                'level': level,
                'propagate': False,
            }
        }
    }
