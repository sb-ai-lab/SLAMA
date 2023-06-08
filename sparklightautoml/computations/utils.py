import logging
import multiprocessing
from copy import deepcopy
from typing import List

from pyspark import SparkContext, inheritable_thread_target, RDD
from pyspark.sql import SparkSession

from sparklightautoml.utils import SparkDataFrame, get_current_session
from sparklightautoml.validation.base import SparkBaseTrainValidIterator

logger = logging.getLogger(__name__)


def get_executors() -> List[str]:
    # noinspection PyUnresolvedReferences
    sc = SparkContext._active_spark_context
    return sc._jvm.org.apache.spark.lightautoml.utils.SomeFunctions.executors()


def get_executors_cores() -> int:
    master_addr = get_current_session().conf.get("spark.master")
    if master_addr.startswith("local-cluster"):
        _, cores_str, _ = master_addr[len("local-cluster["): -1].split(",")
        cores = int(cores_str)
    elif master_addr.startswith("local"):
        cores_str = master_addr[len("local["): -1]
        cores = int(cores_str) if cores_str != "*" else multiprocessing.cpu_count()
    else:
        cores = int(get_current_session().conf.get("spark.executor.cores", "1"))

    return cores


def duplicate_on_num_slots_with_locations_preferences(
        df: SparkDataFrame,
        num_slots: int,
        materialize_base_rdd: bool = True,
        enforce_division_without_reminder: bool = True):
    # noinspection PyUnresolvedReferences
    spark = get_current_session()
    sc = SparkContext._active_spark_context
    result = sc._jvm.org.apache.spark.lightautoml.utils.SomeFunctions.duplicateOnNumSlotsWithLocationsPreferences(
        df._jdf, num_slots, materialize_base_rdd, enforce_division_without_reminder
    )
    dfs = [SparkDataFrame(jobj, spark._wrapped) for jobj in result._1()]
    base_coalesced_df = SparkDataFrame(result._2(), spark._wrapped)
    return dfs, base_coalesced_df


def inheritable_thread_target_with_exceptions_catcher(f):
    def _func():
        try:
            return f()
        except:
            logger.error("Error in a compute thread", exc_info=True)
            raise

    return inheritable_thread_target(_func)


def deecopy_tviter_without_dataset(tv_iter: SparkBaseTrainValidIterator) -> SparkBaseTrainValidIterator:
    train = tv_iter.train
    tv_iter.train = None
    tv_iter_copy = deepcopy(tv_iter)
    tv_iter.train = train
    return tv_iter_copy
