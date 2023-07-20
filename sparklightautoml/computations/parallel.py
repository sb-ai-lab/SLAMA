import math
import warnings

from contextlib import contextmanager
from multiprocessing.pool import ThreadPool
from queue import Queue
from typing import Callable
from typing import List
from typing import Optional

from sparklightautoml.computations.base import ComputationSlot
from sparklightautoml.computations.base import ComputationsManager
from sparklightautoml.computations.base import ComputationsSession
from sparklightautoml.computations.base import R
from sparklightautoml.computations.base import T
from sparklightautoml.computations.base import logger
from sparklightautoml.computations.utils import (
    duplicate_on_num_slots_with_locations_preferences,
)
from sparklightautoml.computations.utils import get_executors
from sparklightautoml.computations.utils import get_executors_cores
from sparklightautoml.computations.utils import (
    inheritable_thread_target_with_exceptions_catcher,
)
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.utils import SparkDataFrame


class ParallelComputationsSession(ComputationsSession):
    def __init__(self, dataset: SparkDataset, parallelism: int, use_location_prefs_mode: int):
        super(ParallelComputationsSession, self).__init__()
        self._parallelism = parallelism
        self._use_location_prefs_mode = use_location_prefs_mode
        self._dataset = dataset
        self._computing_slots: Optional[List[ComputationSlot]] = None
        self._available_computing_slots_queue: Optional[Queue] = None
        self._pool: Optional[ThreadPool] = None
        self._base_pref_locs_df: Optional[SparkDataFrame] = None

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
        if self._base_pref_locs_df is not None:
            self._base_pref_locs_df.unpersist()

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
            lambda task: inheritable_thread_target_with_exceptions_catcher(lambda: func(task))(), tasks
        )

    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        assert self._pool is not None
        # TODO: PARALLEL - probably, is not fully correct and needs to be integrated on the thread pool level,
        #  inlcuding one-shot threads
        return self._pool.map(lambda f: inheritable_thread_target_with_exceptions_catcher(f)(), tasks)

    def _make_computing_slots(self, dataset: Optional[SparkDataset]) -> List[ComputationSlot]:
        if dataset is not None and self._use_location_prefs_mode:
            computing_slots = self._make_slots_on_dataset_copies_coalesced_into_preffered_locations(dataset)
        elif dataset is not None:
            computing_slots = self._make_slots_with_current_dataset(dataset)
        else:
            computing_slots = [ComputationSlot(f"{i}") for i in range(self._parallelism)]

        return computing_slots

    def _make_slots_with_current_dataset(self, dataset: SparkDataset) -> List[ComputationSlot]:
        execs = get_executors()
        exec_cores = get_executors_cores()

        num_tasks = max(1, math.floor(len(execs) * exec_cores / self._parallelism))
        num_threads_per_executor = max(1, round(num_tasks / len(execs)))

        computing_slots = [
            ComputationSlot(f"{i}", dataset, num_tasks=num_tasks, num_threads_per_executor=num_threads_per_executor)
            for i in range(self._parallelism)
        ]

        return computing_slots

    def _make_slots_on_dataset_copies_coalesced_into_preffered_locations(
        self, dataset: SparkDataset
    ) -> List[ComputationSlot]:
        logger.warning(
            "Be aware for correct functioning slot-based computations "
            "there should noy be any parallel computations from "
            "different entities (other MLPipes, MLAlgo, etc)."
        )

        execs = get_executors()
        exec_cores = get_executors_cores()

        if math.floor(len(execs) / self._parallelism) > 0 and len(execs) % self._parallelism != 0:
            warnings.warn(
                f"Uneven number of executors per job (one job uses multiple executors): "
                f"{len(execs)} / {self._parallelism}. "
                f"Number of allocated slots may be reduced."
            )

        if math.floor(len(execs) / self._parallelism) == 0 and exec_cores % self._parallelism != 0:
            warnings.warn(
                f"Uneven number of jobs per executor (one job uses only a part of a single executor): "
                f"{exec_cores} / {self._parallelism}. "
                f"Number of allocated slots may be reduced."
            )

        logger.info(
            f"Coalescing dataset into multiple copies (num copies: {self._parallelism}) "
            f"with specified preffered locations"
        )

        if self._parallelism > 1:
            dfs, self._base_pref_locs_df = duplicate_on_num_slots_with_locations_preferences(
                df=dataset.data, num_slots=self._parallelism, enforce_division_without_reminder=False
            )
        else:
            dfs, self._base_pref_locs_df = [dataset.data], None

        assert len(dfs) > 0, "Not dataframe slots are prepared, cannot continue"

        if len(dfs) != self._parallelism:
            logger.warning(
                f"Impossible to allocate desired number of slots, "
                f"probably due to lack of resources: {len(dfs)} < {self._parallelism}"
            )

        # checking all dataframes have the same number of partitions
        unique_num_partitions = set(df.rdd.getNumPartitions() for df in dfs)
        assert len(unique_num_partitions) == 1 and unique_num_partitions.pop() > 0

        num_tasks = dfs[0].rdd.getNumPartitions()
        num_threads_per_executor = min(num_tasks, exec_cores)

        logger.info(f"Continuing with {len(dfs)} prepared slots.")

        dataset_slots = []
        for i, coalesced_df in enumerate(dfs):
            coalesced_dataset = dataset.empty()
            coalesced_dataset.set_data(
                coalesced_df, dataset.features, dataset.roles, name=f"CoalescedForPrefLocs_{dataset.name}"
            )

            # TODO: PARALLEL - add preffered locations logging
            dataset_slots.append(
                ComputationSlot(
                    id=f"{i}",
                    dataset=coalesced_dataset,
                    num_tasks=num_tasks,
                    num_threads_per_executor=num_threads_per_executor,
                )
            )

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
