import logging
import math
import multiprocessing
import threading
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from multiprocessing.pool import ThreadPool
from typing import Callable, Optional, Iterable, Any, Dict, Tuple, cast
from typing import TypeVar, List

from pyspark import inheritable_thread_target, SparkContext, keyword_only
from pyspark.ml.common import inherit_doc
from pyspark.ml.wrapper import JavaTransformer
from pyspark.sql import SparkSession

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.validation.base import SparkBaseTrainValidIterator, TrainVal

logger = logging.getLogger(__name__)

ENV_VAR_SLAMA_COMPUTATIONS_MANAGER = "SLAMA_COMPUTATIONS_MANAGER"

T = TypeVar("T")
S = TypeVar("S", bound='Slot')


class PoolType(Enum):
    ml_pipelines = "ml_pipelines"
    ml_algos = "ml_algos"
    job = "job"


__pools__: Optional[Dict[PoolType, ThreadPool]] = None


def create_pools(ml_pipes_pool_size: int = 1, ml_algos_pool_size: int = 1, job_pool_size: int = 1):
    global __pools__

    assert __pools__ is None, "Cannot recreate already existing thread pools"

    __pools__ = {
        PoolType.ml_pipelines: ThreadPool(processes=ml_pipes_pool_size) if ml_pipes_pool_size > 1 else None,
        PoolType.ml_algos: ThreadPool(processes=ml_algos_pool_size) if ml_algos_pool_size > 1 else None,
        PoolType.job: ThreadPool(processes=job_pool_size) if job_pool_size > 1 else None
    }


def get_pool(pool_type: PoolType) -> Optional[ThreadPool]:
    return __pools__.get(pool_type, None)


# noinspection PyUnresolvedReferences
def get_executors() -> List[str]:
    sc = SparkContext._active_spark_context
    return sc._jvm.org.apache.spark.lightautoml.utils.SomeFunctions.get_executors()


def get_executors_cores() -> int:
    master_addr = SparkSession.getActiveSession().conf.get("spark.master")
    if master_addr.startswith("local-cluster"):
        _, cores_str, _ = master_addr[len("local-cluster["): -1].split(",")
        cores = int(cores_str)
        num_threads = max(cores - 1, 1)
    elif master_addr.startswith("local"):
        cores_str = master_addr[len("local["): -1]
        cores = int(cores_str) if cores_str != "*" else multiprocessing.cpu_count()
        num_threads = max(cores - 1, 1)
    else:
        num_threads = max(int(SparkSession.getActiveSession().conf.get("spark.executor.cores", "1")) - 1, 1)

    return num_threads


def _compute_sequential(tasks: List[Callable[[], T]]) -> List[T]:
    return [task() for task in tasks]


def build_named_parallelism_settings(config_name: str, parallelism: int):
    parallelism_config = {
        "no_parallelism": None,
        "intra_mlpipe_parallelism": {
            PoolType.ml_pipelines.name: 1,
            PoolType.ml_algos.name: 1,
            PoolType.job.name: parallelism,
            "tuner": parallelism,
            "linear_l2": {},
            "lgb": {
                "parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "experimental_parallel_mode": False
            }
        },
        "intra_mlpipe_parallelism_with_experimental_features": {
            PoolType.ml_pipelines.name: 1,
            PoolType.ml_algos.name: 1,
            PoolType.job.name: parallelism,
            "tuner": parallelism,
            "linear_l2": {},
            "lgb": {
                "parallelism": parallelism,
                "use_barrier_execution_mode": True,
                "experimental_parallel_mode": True
            }
        },
        "mlpipe_level_parallelism": {
            PoolType.ml_pipelines.name: parallelism,
            PoolType.ml_algos.name: 1,
            PoolType.job.name: 1,
            "tuner": 1,
            "linear_l2": {},
            "lgb": {}
        }
    }

    assert config_name in parallelism_config, \
        f"Not supported parallelism mode: {config_name}. " \
        f"Only the following ones are supoorted at the moment: {list(parallelism_config.keys())}"

    return parallelism_config[config_name]


@inherit_doc
class PrefferedLocsPartitionCoalescerTransformer(JavaTransformer):
    """
    Custom implementation of PySpark BalancedUnionPartitionsCoalescerTransformer wrapper
    """

    @keyword_only
    def __init__(self, pref_locs: List[str], do_shuffle: bool = True):
        super(PrefferedLocsPartitionCoalescerTransformer, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.lightautoml.utils.PrefferedLocsPartitionCoalescerTransformer",
            self.uid, pref_locs, do_shuffle
        )


@dataclass
class DatasetSlot:
    dataset: SparkDataset
    free: bool


@dataclass
class SlotSize:
    num_tasks: int
    num_threads_per_executor: int


class SlotAllocator(ABC):
    @abstractmethod
    @contextmanager
    def allocate(self) -> DatasetSlot:
        ...

    @property
    @abstractmethod
    def slot_size(self) -> SlotSize:
        ...


class ParallelSlotAllocator(SlotAllocator):
    def __init__(self, slot_size: SlotSize, slots: List[DatasetSlot], pool: ThreadPool):
        self._slot_size = slot_size
        self._slots = slots
        self._pool = pool
        self._slots_lock = threading.Lock()

    @property
    def slot_size(self) -> SlotSize:
        return self._slot_size

    @contextmanager
    def allocate(self) -> DatasetSlot:
        with self._slots_lock:
            free_slot = next((slot for slot in self._slots if slot.free))
            free_slot.free = False

        yield free_slot

        with self._slots_lock:
            free_slot.free = True


class ComputationsManager(ABC):
    @abstractmethod
    def compute(self, tasks: List[Callable[[], T]], pool_type: PoolType = PoolType.job) -> List[T]:
        ...

    @property
    @abstractmethod
    def default_slot_size(self) -> Optional[SlotSize]:
        ...

    @property
    @abstractmethod
    def can_support_slots(self) -> bool:
        ...

    @contextmanager
    @abstractmethod
    def slots(self, dataset: SparkDataset, parallelism: int, pool_type: PoolType) -> SlotAllocator:
        ...


class ParallelComputationsManager(ComputationsManager):
    def __init__(self,
                 ml_pipes_pool_size: int = 1,
                 ml_algos_pool_size: int = 1,
                 job_pool_size: int = 1):
        # doing it, because ParallelComputations Manager should be deepcopy-able
        create_pools(ml_pipes_pool_size, ml_algos_pool_size, job_pool_size)
        self._default_slot_size = None

    def _get_pool(self, pool_type: PoolType) -> Optional[ThreadPool]:
        return get_pool(pool_type)

    @property
    def can_support_slots(self) -> bool:
        return True

    @property
    def default_slot_size(self) -> Optional[SlotSize]:
        return self._default_slot_size

    def compute(self, tasks: List[Callable[[], T]], pool_type: PoolType = PoolType.job) -> List[T]:
        pool = self._get_pool(pool_type)

        if not pool:
            return _compute_sequential(tasks)

        ptasks = map(inheritable_thread_target, tasks)
        results = sorted(
            (result for result in pool.imap_unordered(lambda f: f(), ptasks) if result),
            key=lambda x: x[0]
        )

        return results

    @contextmanager
    def slots(self, dataset: SparkDataset, parallelism: int, pool_type: PoolType) -> SlotAllocator:
        logger.warning("Beaware for correct functioning slot-based computations "
                       "there should noy be any parallel computations from "
                       "different entities (other MLPipes, MLAlgo, etc).")

        pool = self._get_pool(pool_type)
        slot_size, slots = self._prepare_trains(dataset=dataset, parallelism=parallelism)

        yield ParallelSlotAllocator(slot_size, slots, pool)

        logger.info("Clear cache of dataset copies (slots) for the coalesced dataset")
        for slot in slots:
            slot.dataset.data.unpersist()

    @staticmethod
    def _prepare_trains(dataset: SparkDataset, parallelism: int) -> Tuple[SlotSize, List[DatasetSlot]]:
        execs = get_executors()
        exec_cores = get_executors_cores(dataset.data)
        execs_per_slot = max(1, math.floor(len(execs) / parallelism))
        slots_num = int(len(execs) / execs_per_slot)
        slot_size = SlotSize(num_tasks=execs_per_slot * exec_cores, num_threads_per_executor=exec_cores)

        if len(execs) % parallelism != 0:
            warnings.warn(f"Uneven number of executors per job. "
                          f"Setting execs per slot: {execs_per_slot}, slots num: {slots_num}.")

        logger.info(f"Coalescing dataset into multiple copies (num copies: {slots_num}) "
                    f"with specified preffered locations")

        _train_slots = []

        for i in range(slots_num):
            pref_locs = execs[i * execs_per_slot: (i + 1) * execs_per_slot]

            coalesced_data = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs)\
                .transform(dataset.data).cache()
            coalesced_data.write.mode('overwrite').format('noop').save()

            coalesced_dataset = dataset.empty()
            coalesced_dataset.set_data(coalesced_data, coalesced_dataset.features, coalesced_dataset.roles,
                                       name=f"CoalescedForPrefLocs_{dataset.name}")

            _train_slots.append(DatasetSlot(dataset=coalesced_dataset,free=True))

            logger.info(f"Preffered locations for slot #{i}: {pref_locs}")

        return slot_size, _train_slots


class SequentialComputationsManager(ComputationsManager):
    def __init__(self, default_slot_size: Optional[SlotSize] = None):
        super(SequentialComputationsManager, self).__init__()
        self._default_slot_size = default_slot_size

    @property
    def default_slot_size(self) -> Optional[SlotSize]:
        return self._default_slot_size

    @property
    def can_support_slots(self) -> bool:
        return False

    def slots(self, dataset: SparkDataset, parallelism: int, pool_type: PoolType) -> SlotAllocator:
        raise NotImplementedError("Not supported by this computational manager")

    def compute(self, tasks: list[Callable[[], T]], pool_type: PoolType = PoolType.job) -> List[T]:
        return _compute_sequential(tasks)


def build_computations_manager(parallelism_settings: Optional[Dict[str, Any]] = None):
    if parallelism_settings is None:
        comp_manager = SequentialComputationsManager()
    else:
        comp_manager = ParallelComputationsManager(
            ml_pipes_pool_size=parallelism_settings[PoolType.ml_pipelines.name],
            ml_algos_pool_size=parallelism_settings[PoolType.ml_algos.name],
            job_pool_size=parallelism_settings[PoolType.job.name]
        )
    return comp_manager


def default_computations_manager() -> ComputationsManager:
    return SequentialComputationsManager()


def compute_tasks(tasks: List[Callable[[], T]], pool_type: PoolType = PoolType.job) -> List[T]:
    return default_computations_manager().compute(tasks, pool_type)


class _SlotBasedTVIter(SparkBaseTrainValidIterator):
    def __init__(self, slots: Callable[[], DatasetSlot], tviter: SparkBaseTrainValidIterator):
        super().__init__(None)
        self._slots = slots
        self._tviter = tviter
        self._curr_pos = 0

    def __iter__(self) -> Iterable:
        self._curr_pos = 0
        return self

    def __len__(self) -> Optional[int]:
        return len(self._tviter)

    def __next__(self) -> TrainVal:
        with self._slots() as slot:
            tviter = deepcopy(self._tviter)
            tviter.train = slot.dataset

            self._curr_pos += 1

            try:
                curr_tv = None
                for i in range(self._curr_pos):
                    curr_tv = next(tviter)
            except StopIteration:
                self._curr_pos = 0
                raise StopIteration()

        return curr_tv

    def convert_to_holdout_iterator(self):
        return _SlotBasedTVIter(
            self._slots,
            cast(SparkBaseTrainValidIterator, self._tviter.convert_to_holdout_iterator())
        )

    def freeze(self) -> 'SparkBaseTrainValidIterator':
        raise NotImplementedError()

    def unpersist(self, skip_val: bool = False):
        raise NotImplementedError()

    def get_validation_data(self) -> SparkDataset:
        return self._tviter.get_validation_data()


class _SlotInitiatedTVIter(SparkBaseTrainValidIterator):
    def __len__(self) -> Optional[int]:
        return len(self._tviter)

    def convert_to_holdout_iterator(self):
        return _SlotInitiatedTVIter(self._slot_allocator, self._tviter.convert_to_holdout_iterator())

    def __init__(self, slot_allocator: SlotAllocator, tviter: SparkBaseTrainValidIterator):
        super().__init__(None)
        self._slot_allocator = slot_allocator
        self._tviter = deepcopy(tviter)

    def __iter__(self) -> Iterable:
        def _iter():
            with self._slot_allocator.allocate() as slot:
                tviter = deepcopy(self._tviter)
                tviter.train = slot.dataset
                for elt in tviter:
                    yield elt

        return _iter()

    def __next__(self):
        raise NotImplementedError("NotSupportedMethod")

    def freeze(self) -> 'SparkBaseTrainValidIterator':
        raise NotImplementedError("NotSupportedMethod")

    def unpersist(self, skip_val: bool = False):
        raise NotImplementedError("NotSupportedMethod")

    def get_validation_data(self) -> SparkDataset:
        return self._tviter.get_validation_data()