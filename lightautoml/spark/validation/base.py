"""Basic classes for validation iterators."""

from copy import copy
from typing import Any, Generator, Iterable, List, Optional, Sequence, Tuple, TypeVar, cast

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.pipelines.features.base import FeaturesPipeline


# from ..pipelines.selection.base import SelectionPipeline

# TODO: SOLVE CYCLIC IMPORT PROBLEM!!! add Selectors typing

# Dataset = TypeVar("Dataset", bound=LAMLDataset)
CustomIdxs = Iterable[Tuple[Sequence, Sequence]]


# add checks here
# check for same columns in dataset
class TrainValidIterator:
    """Abstract class to train/validation iteration.

    Train/valid iterator:
    should implement `__iter__` and `__next__` for using in ml_pipeline.

    """

    @property
    def features(self):
        return self.train.features

    def __init__(self, train: SparkDataset, **kwargs: Any):
        self.train = train
        for k in kwargs:
            self.__dict__[k] = kwargs[k]

    def __iter__(self) -> Iterable:
        """ Abstract method. Creates iterator."""
        raise NotImplementedError

    def __len__(self) -> Optional[int]:
        """Abstract method. Get length of dataset."""
        raise NotImplementedError

    def get_validation_data(self) -> SparkDataset:
        """Abstract method. Get validation sample."""
        raise NotImplementedError

    def apply_feature_pipeline(self, features_pipeline: FeaturesPipeline) -> "TrainValidIterator":

        train_valid = copy(self)
        train_valid.train = features_pipeline.fit_transform(train_valid.train)
        return train_valid

    # TODO: add typing
    def apply_selector(self, selector) -> "TrainValidIterator":

        if not selector.is_fitted:
            selector.fit(self)
        train_valid = copy(self)
        train_valid.train = selector.select(train_valid.train)
        return train_valid

    def convert_to_holdout_iterator(self) -> "HoldoutIterator":
        """Abstract method. Convert iterator to HoldoutIterator."""
        raise NotImplementedError


class DummyIterator(TrainValidIterator):
    """Simple Iterator which use train data as validation."""

    def __init__(self, train: SparkDataset):

        self.train = train
        self.valid = train

    def __len__(self) -> Optional[int]:

        return 1

    def __iter__(self) -> List[Tuple[None, SparkDataset, SparkDataset]]:

        return [(None, self.train, self.train)]

    def get_validation_data(self) -> SparkDataset:

        return self.train

    def convert_to_holdout_iterator(self) -> "HoldoutIterator":

        return HoldoutIterator(self.train, self.train)


class HoldoutIterator(TrainValidIterator):

    def __init__(self, train: SparkDataset, valid: SparkDataset):

        self.train = train
        self.valid = valid

    def __len__(self) -> Optional[int]:

        return 1

    def __iter__(self) -> Iterable[Tuple[None, SparkDataset, SparkDataset]]:

        return iter([(None, self.train, self.valid)])

    def get_validation_data(self) -> SparkDataset:

        return self.valid

    def apply_feature_pipeline(self, features_pipeline: FeaturesPipeline) -> "HoldoutIterator":

        train_valid = cast("HoldoutIterator", super().apply_feature_pipeline(features_pipeline))
        train_valid.valid = features_pipeline.transform(train_valid.valid)

        return train_valid

    def apply_selector(self, selector) -> "HoldoutIterator":
        """Same as for basic class, but also apply to validation.

        Args:
            selector: Uses for feature selection.

        Returns:
            New iterator.

        """
        train_valid = cast("HoldoutIterator", super().apply_selector(selector))
        train_valid.valid = selector.select(train_valid.valid)

        return train_valid

    def convert_to_holdout_iterator(self) -> "HoldoutIterator":
        """Do nothing, just return itself.

        Returns:
            self.

        """
        return self


class CustomIterator(TrainValidIterator):
    """Iterator that uses function to create folds indexes.

    Usefull for example - classic timeseries splits.

    """

    def __init__(self, train: SparkDataset, iterator: CustomIdxs):
        """Create iterator.

        Args:
            train: Dataset of train data.
            iterator: Callable(dataset) -> Iterator of train/valid indexes.

        """
        self.train = train
        self.iterator = iterator

    def __len__(self) -> Optional[int]:
        """Empty __len__ method.

        Returns:
            None.

        """

        return len(self.iterator)

    def __iter__(self) -> Generator:
        """Create generator of train/valid datasets.

        Returns:
            Data generator.

        """
        generator = ((val_idx, self.train[tr_idx], self.train[val_idx]) for (tr_idx, val_idx) in self.iterator)

        return generator

    def get_validation_data(self) -> SparkDataset:
        """Simple return train dataset.

        Returns:
            Dataset of train data.

        """
        return self.train

    def convert_to_holdout_iterator(self) -> "HoldoutIterator":
        """Convert iterator to hold-out-iterator.

        Use first train/valid split for :class:`~lightautoml.validation.base.HoldoutIterator` creation.

        Returns:
            New hold out iterator.

        """
        for (tr_idx, val_idx) in self.iterator:
            return HoldoutIterator(self.train[tr_idx], self.train[val_idx])
