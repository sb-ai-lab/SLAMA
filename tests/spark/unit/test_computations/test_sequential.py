import collections
import threading
from copy import deepcopy

import pytest
from pyspark.sql import SparkSession

from sparklightautoml.computations.sequential import SequentialComputationsManager
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.validation.iterators import SparkFoldsIterator
from . import build_func, build_idx_func, TestWorkerException, build_func_on_dataset, build_fold_func, \
    build_func_with_exception
from .. import spark as spark_sess, dataset as spark_dataset

spark = spark_sess
dataset = spark_dataset

K = 20


@pytest.mark.parametrize("num_tasks,num_threads", [(None, None), (None, 2), (4, None), (2, 4), (4, 2)])
def test_allocate(spark: SparkSession, dataset: SparkDataset, num_tasks, num_threads):
    manager = SequentialComputationsManager(num_tasks=num_tasks, num_threads_per_executor=num_threads)

    with manager.session(dataset) as session:
        for i in range(10):
            with session.allocate() as slot:
                assert slot.dataset is not None
                assert slot.dataset.uid == dataset.uid
                assert slot.num_tasks == num_tasks
                assert slot.num_threads_per_executor == num_threads

        acc = collections.deque()
        results = session.compute([build_func(acc, j) for j in range(K)])
        unique_thread_ids = set(acc)

        assert results == list(range(K))
        assert len(unique_thread_ids) == 1
        assert next(iter(unique_thread_ids)) == threading.get_ident()

        acc = collections.deque()
        results = session.map_and_compute(build_idx_func(acc), list(range(K)))
        unique_thread_ids = set(acc)
        assert results == list(range(K))
        assert len(unique_thread_ids) == 1
        assert next(iter(unique_thread_ids)) == threading.get_ident()


def test_compute():
    acc = collections.deque()
    tasks = [build_func(acc, i) for i in range(K)]

    manager = SequentialComputationsManager()
    results = manager.compute(tasks)
    unique_thread_ids = set(acc)

    assert results == list(range(K))
    assert len(unique_thread_ids) == 1
    assert next(iter(unique_thread_ids)) == threading.get_ident()


def test_compute_with_exceptions(spark: SparkSession):
    acc = collections.deque()
    tasks = [*(build_func_with_exception(acc, i) for i in range(K, K + 3)), *(build_func(acc, i) for i in range(K))]

    manager = SequentialComputationsManager()
    with pytest.raises(TestWorkerException):
        manager.compute(tasks)


def test_compute_on_dataset(spark: SparkSession, dataset: SparkDataset):
    acc = collections.deque()
    tasks = [build_func_on_dataset(acc, i) for i in range(K)]

    manager = SequentialComputationsManager()
    results = manager.compute_on_dataset(dataset, tasks)
    unique_thread_ids = set(acc)

    assert results == list(range(K))
    assert len(unique_thread_ids) == 1
    assert next(iter(unique_thread_ids)) == threading.get_ident()


def test_compute_on_train_val_iter(spark: SparkSession, dataset: SparkDataset):
    n_folds = dataset.num_folds
    acc = collections.deque()
    tv_iter = SparkFoldsIterator(dataset)
    task = build_fold_func(acc)

    manager = SequentialComputationsManager()
    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == 1
    assert next(iter(unique_thread_ids)) == threading.get_ident()


def test_deepcopy(spark: SparkSession, dataset: SparkDataset):
    n_folds = dataset.num_folds
    acc = collections.deque()
    tv_iter = SparkFoldsIterator(dataset)
    task = build_fold_func(acc)

    manager = SequentialComputationsManager()

    manager = deepcopy(manager)

    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == 1
    assert next(iter(unique_thread_ids)) == threading.get_ident()

    manager = deepcopy(manager)

    results = manager.compute_folds(tv_iter, task)
    unique_thread_ids = set(acc)

    assert results == list(range(n_folds))
    assert len(unique_thread_ids) == 1
    assert next(iter(unique_thread_ids)) == threading.get_ident()
