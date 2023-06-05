import math
import warnings
from contextlib import contextmanager
from multiprocessing.pool import ThreadPool
from queue import Queue
from typing import Optional, List, Callable

from sparklightautoml.computations.base import ComputationsSession, ComputationSlot, T, R, logger, \
    ComputationsManager
from sparklightautoml.computations.utils import inheritable_thread_target_with_exceptions_catcher, get_executors, \
    get_executors_cores
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.transformers.scala_wrappers.preffered_locs_partition_coalescer import \
    PrefferedLocsPartitionCoalescerTransformer


class ParallelComputationsSession(ComputationsSession):
    def __init__(self, dataset: SparkDataset, parallelism: int, use_location_prefs_mode: int):
        super(ParallelComputationsSession, self).__init__()
        self._parallelism = parallelism
        self._use_location_prefs_mode = use_location_prefs_mode
        self._dataset = dataset
        self._computing_slots: Optional[List[ComputationSlot]] = None
        self._available_computing_slots_queue: Optional[Queue] = None
        self._pool: Optional[ThreadPool] = None

    def __enter__(self):
        self._pool = ThreadPool(processes=self._parallelism)
        self._computing_slots = self._make_computing_slots(self._dataset)
        self._available_computing_slots_queue = Queue(maxsize=len(self._computing_slots))
        for cslot in self._computing_slots:
            self._available_computing_slots_queue.put(cslot)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.terminate()
        self._pool = None
        self._available_computing_slots_queue = None
        if self._computing_slots is not None:
            for cslot in self._computing_slots:
                if cslot.dataset is not None:
                    cslot.dataset.unpersist()

    @contextmanager
    def allocate(self) -> ComputationSlot:
        slot = None
        try:
            assert self._available_computing_slots_queue is not None, "Cannot allocate slots without session"
            slot = self._available_computing_slots_queue.get()
            yield slot
        finally:
            if slot is not None:
                self._available_computing_slots_queue.put(slot)

    def map_and_compute(self, func: Callable[[R], T], tasks: List[R]) -> List[T]:
        assert self._pool is not None
        # TODO: PARALLEL - probably, is not fully correct and needs to be integrated on the thread pool level, 
        #  inlcuding one-shot threads
        return self._pool.map(
            lambda task: inheritable_thread_target_with_exceptions_catcher(lambda: func(task))(), 
            tasks
        )

    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        assert self._pool is not None
        # TODO: PARALLEL - probably, is not fully correct and needs to be integrated on the thread pool level, 
        #  inlcuding one-shot threads
        return self._pool.map(
            lambda f: inheritable_thread_target_with_exceptions_catcher(f)(), 
            tasks
        )

    def _make_computing_slots(self, dataset: Optional[SparkDataset]) -> List[ComputationSlot]:
        if dataset is not None and self._use_location_prefs_mode:
            computing_slots = self._coalesced_dataset_copies_into_preffered_locations(dataset)
        else:
            computing_slots = [ComputationSlot(f"i", dataset) for i in range(self._parallelism)]
        return computing_slots

    def _coalesced_dataset_copies_into_preffered_locations(self, dataset: SparkDataset) \
            -> List[ComputationSlot]:
        logger.warning("Be aware for correct functioning slot-based computations "
                       "there should noy be any parallel computations from "
                       "different entities (other MLPipes, MLAlgo, etc).")

        # TODO: PARALLEL - improve function to work with uneven number of executors
        execs = get_executors()
        exec_cores = get_executors_cores()
        execs_per_slot = max(1, math.floor(len(execs) / self._parallelism))
        slots_num = int(len(execs) / execs_per_slot)
        num_tasks = execs_per_slot * exec_cores
        num_threads_per_executor = max(exec_cores - 1, 1)

        if len(execs) % self._parallelism != 0:
            warnings.warn(f"Uneven number of executors per job. "
                          f"Setting execs per slot: {execs_per_slot}, slots num: {slots_num}.")

        logger.info(f"Coalescing dataset into multiple copies (num copies: {slots_num}) "
                    f"with specified preffered locations")

        dataset_slots = []

        # TODO: PARALLEL - may be executed in parallel
        # TODO: PARALLEL - it might be optimized on Scala level and squashed into a single operation
        for i in range(slots_num):
            pref_locs = execs[i * execs_per_slot: (i + 1) * execs_per_slot]

            coalesced_data = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs) \
                .transform(dataset.data).cache()
            coalesced_data.write.mode('overwrite').format('noop').save()

            coalesced_dataset = dataset.empty()
            coalesced_dataset.set_data(coalesced_data, dataset.features, dataset.roles,
                                       name=f"CoalescedForPrefLocs_{dataset.name}")

            dataset_slots.append(ComputationSlot(
                id=f"{i}",
                dataset=coalesced_dataset,
                num_tasks=num_tasks,
                num_threads_per_executor=num_threads_per_executor
            ))

            logger.debug(f"Preffered locations for slot #{i}: {pref_locs}")

        return dataset_slots


class ParallelComputationsManager(ComputationsManager):
    def __init__(self, parallelism: int = 1, use_location_prefs_mode: bool = False):
        assert parallelism >= 1
        self._parallelism = parallelism
        self._use_location_prefs_mode = use_location_prefs_mode

    @property
    def parallelism(self) -> int:
        return self._parallelism

    def session(self, dataset: Optional[SparkDataset] = None) -> ParallelComputationsSession:
        return ParallelComputationsSession(dataset, self._parallelism, self._use_location_prefs_mode)
