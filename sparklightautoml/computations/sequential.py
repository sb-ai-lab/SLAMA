from contextlib import contextmanager
from typing import Callable
from typing import List
from typing import Optional

from sparklightautoml.computations.base import ComputationSlot
from sparklightautoml.computations.base import ComputationsManager
from sparklightautoml.computations.base import ComputationsSession
from sparklightautoml.computations.base import R
from sparklightautoml.computations.base import T
from sparklightautoml.dataset.base import SparkDataset


class SequentialComputationsSession(ComputationsSession):
    def __init__(
        self,
        dataset: Optional[SparkDataset] = None,
        num_tasks: Optional[int] = None,
        num_threads_per_executor: Optional[int] = None,
    ):
        super(SequentialComputationsSession, self).__init__()
        self._dataset = dataset
        self._num_tasks = num_tasks
        self._num_threads_per_executor = num_threads_per_executor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @contextmanager
    def allocate(self) -> ComputationSlot:
        yield ComputationSlot(
            "0", self._dataset, num_tasks=self._num_tasks, num_threads_per_executor=self._num_threads_per_executor
        )

    def map_and_compute(self, func: Callable[[R], T], tasks: List[R]) -> List[T]:
        return [func(task) for task in tasks]

    def compute(self, tasks: List[Callable[[], T]]) -> List[T]:
        return [task() for task in tasks]


class SequentialComputationsManager(ComputationsManager):
    def __init__(self, num_tasks: Optional[int] = None, num_threads_per_executor: Optional[int] = None):
        super(SequentialComputationsManager, self).__init__()
        self._dataset: Optional[SparkDataset] = None
        self._num_tasks = num_tasks
        self._num_threads_per_executor = num_threads_per_executor

    @property
    def parallelism(self) -> int:
        return 1

    def session(self, dataset: Optional[SparkDataset] = None) -> SequentialComputationsSession:
        return SequentialComputationsSession(
            dataset, num_tasks=self._num_tasks, num_threads_per_executor=self._num_threads_per_executor
        )
