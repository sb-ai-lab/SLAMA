import collections
import itertools
from copy import deepcopy

import pytest
from pyspark.sql import SparkSession

from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.validation.iterators import SparkFoldsIterator
from . import build_func, build_idx_func, build_func_with_exception, TestWorkerException, build_func_on_dataset, \
    build_fold_func
from .. import dataset as spark_dataset, spark_for_function

spark = spark_for_function
dataset = spark_dataset

K = 20

manager_configs = list(itertools.product([1, 2, 5], [False, True]))


@pytest.mark.parametrize("parallelism,use_location_prefs_mode", manager_configs)
def test_allocate(spark: SparkSession, dataset: SparkDataset, parallelism: int, use_location_prefs_mode: bool):
    manager = ParallelComputationsManager(parallelism, use_location_prefs_mode)

    # TODO: add checking for access from different threads
    with manager.session(dataset) as session:
        for i in range(10):
            with session.allocate() as slot:
                assert slot.dataset is not None
                if use_location_prefs_mode:
                    assert slot.dataset.uid != dataset.uid
                else:
                    assert slot.dataset.uid == dataset.uid

        acc = collections.deque()
        results = session.compute([build_func(acc, j) for j in range(K)])
        unique_thread_ids = set(acc)

        assert results == list(range(K))
        assert len(unique_thread_ids) == min(K, parallelism)

        acc = collections.deque()
        results = session.map_and_compute(build_idx_func(acc), list(range(K)))
        unique_thread_ids = set(acc)

        assert results == list(range(K))
        assert len(unique_thread_ids) == min(K, parallelism)


@pytest.mark.parametrize("parallelism,use_location_prefs_mode", manager_configs)
def test_compute(spark: SparkSession, parallelism: int, use_location_prefs_mode: bool):
    acc = collections.deque()
    tasks = [build_func(acc, i, delay=0.25) for i in range(K)]

    manager = ParallelComputationsManager(parallelism, use_location_prefs_mode)
    results = manager.compute(tasks)
    unique_thread_ids = set(acc)

    assert results == list(range(K))
    assert len(unique_thread_ids) == min(K, parallelism)


@pytest.mark.parametrize("parallelism,use_location_prefs_mode", manager_configs)
def test_compute_with_exceptions(spark: SparkSession, parallelism: int, use_location_prefs_mode: bool):
    acc = collections.deque()
    tasks = [*(build_func_with_exception(acc, i) for i in range(K, K + 3)), *(build_func(acc, i) for i in range(K))]

    manager = ParallelComputationsManager(parallelism, use_location_prefs_mode)
    with pytest.raises(TestWorkerException):
        manager.compute(tasks)


@pytest.mark.parametrize("parallelism,use_location_prefs_mode", manager_configs)
def test_compute_on_dataset(spark: SparkSession, dataset: SparkDataset,
                            parallelism: int, use_location_prefs_mode: bool):
    acc = collections.deque()
    tasks = [build_func_on_dataset(acc, i, use_location_prefs_mode, base_dataset=dataset) for i in range(K)]

    manager = ParallelComputationsManager(parallelism, use_location_prefs_mode)
    results = manager.compute_on_dataset(dataset, tasks)
    unique_thread_ids = set(acc)

    assert results == list(range(K))
    assert len(unique_thread_ids) == min(K, parallelism)


@pytest.mark.parametrize("parallelism,use_location_prefs_mode", manager_configs)
def test_compute_on_train_val_iter(spark: SparkSession, dataset: SparkDataset,
                                   parallelism: int, use_location_prefs_mode: bool):
    n_folds = dataset.num_folds
    acc = collections.deque()
    tv_iter = SparkFoldsIterator(dataset)
    task = build_fold_func(acc, base_dataset=dataset)

    manager = ParallelComputationsManager(parallelism, use_location_prefs_mode)
    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == min(n_folds, parallelism)


def test_deepcopy(spark: SparkSession, dataset: SparkDataset):
    n_folds = dataset.num_folds
    parallelism = 5
    use_location_prefs_mode = True
    tv_iter = SparkFoldsIterator(dataset)

    manager = ParallelComputationsManager(parallelism=parallelism, use_location_prefs_mode=use_location_prefs_mode)

    manager = deepcopy(manager)

    acc = collections.deque()
    task = build_fold_func(acc, base_dataset=dataset)
    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == min(n_folds, parallelism)

    manager = deepcopy(manager)

    acc = collections.deque()
    task = build_fold_func(acc, base_dataset=dataset)
    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == min(n_folds, parallelism)
