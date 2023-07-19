import pytest

from sparklightautoml.computations import parallel
from sparklightautoml.computations.parallel import ParallelComputationsSession
from sparklightautoml.dataset.base import SparkDataset
from .. import dataset as spark_dataset, spark_for_function

spark = spark_for_function
dataset = spark_dataset


@pytest.mark.parametrize("exec_nums,exec_cores,parallelism,correct_num_tasks,correct_num_threads", [
    (8, 4, 1, 32, 4),
    (8, 4, 2, 16, 2),
    (8, 4, 4, 8, 1),
    (8, 4, 8, 4, 1),
    (8, 4, 16, 2, 1),
    (8, 4, 32, 1, 1),
    (8, 4, 64, 1, 1),

    (4, 8, 1, 32, 8),
    (4, 8, 2, 16, 4),
    (4, 8, 4, 8, 2),
    (4, 8, 8, 4, 1),
    (4, 8, 16, 2, 1),
    (4, 8, 32, 1, 1),
    (4, 8, 64, 1, 1),

    (7, 3, 1, 21, 3),
    (7, 3, 2, 10, 1),
    (7, 3, 4, 5, 1),
    (7, 3, 8, 2, 1),
    (7, 3, 16, 1, 1),
    (7, 3, 32, 1, 1),
    (7, 3, 64, 1, 1)
])
def test_parallel_computation_session(
        exec_nums: int,
        exec_cores: int,
        parallelism: int,
        correct_num_tasks: int,
        correct_num_threads: int,
        dataset: SparkDataset,
        monkeypatch
):
    monkeypatch.setattr(parallel, "get_executors", lambda: [f"pseudo_exec_{i}" for i in range(exec_nums)])
    monkeypatch.setattr(parallel, "get_executors_cores", lambda: exec_cores)

    with ParallelComputationsSession(dataset, parallelism=parallelism, use_location_prefs_mode=False) as session:
        with session.allocate() as slot:
            assert slot.num_tasks == correct_num_tasks
            assert slot.num_threads_per_executor == correct_num_threads
