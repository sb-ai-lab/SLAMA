import collections
import threading
from typing import Optional

from sparklightautoml.computations.base import ComputationSlot
from sparklightautoml.dataset.base import SparkDataset


class TestWorkerException(Exception):
    def __init__(self, id: int):
        super(TestWorkerException, self).__init__(f"Intentional exception in task {id}")


def build_func(acc: collections.deque, seq_id: int):
    def _func() -> int:
        acc.append(threading.get_ident())
        return seq_id
    return _func


def build_func_with_exception(acc: collections.deque, seq_id: int):
    def _func():
        acc.append(threading.get_ident())
        raise TestWorkerException(seq_id)
    return _func


def build_func_on_dataset(
        acc: collections.deque,
        seq_id: int,
        use_location_prefs_mode: bool = False,
        base_dataset: Optional[SparkDataset] = None
):
    def _func(slot: ComputationSlot) -> int:
        assert slot.dataset is not None
        if base_dataset is not None:
            if use_location_prefs_mode:
                assert slot.dataset.uid != base_dataset.uid
            else:
                assert slot.dataset.uid == base_dataset.uid
            assert slot.dataset.data.count() == base_dataset.data.count()
            assert slot.dataset.data.columns == base_dataset.data.columns
            assert slot.dataset.features == base_dataset.features
            assert slot.dataset.roles == base_dataset.roles
        acc.append(threading.get_ident())
        return seq_id
    return _func


def build_fold_func(acc: collections.deque, base_dataset: Optional[SparkDataset] = None):
    def _func(fold_id: int, slot: ComputationSlot) -> int:
        assert slot.dataset is not None
        if base_dataset is not None:
            assert slot.dataset.uid != base_dataset.uid
            assert slot.dataset.data.count() > 0
            assert [c for c in slot.dataset.data.columns if c != 'is_val'] == base_dataset.data.columns
            assert slot.dataset.features == base_dataset.features
            assert slot.dataset.roles == base_dataset.roles
        acc.append(threading.get_ident())
        return fold_id
    return _func


def build_idx_func(acc: collections.deque):
    def _func(idx: int) -> int:
        acc.append(threading.get_ident())
        return idx
    return _func
