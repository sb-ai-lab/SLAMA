import functools
import logging
from typing import Optional, cast, Iterable, Sequence

from pyspark.sql import functions as sf

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.pipelines.features.base import SparkFeaturesPipeline
from sparklightautoml.utils import SparkDataFrame
from sparklightautoml.validation.base import SparkBaseTrainValidIterator, TrainVal, SparkSelectionPipeline

logger = logging.getLogger(__name__)


class SparkDummyIterator(SparkBaseTrainValidIterator):
    """
    Simple one step iterator over train part of SparkDataset
    """

    def __init__(self, train: SparkDataset):
        super().__init__(train)
        self._curr_idx = 0

    def __iter__(self) -> Iterable:
        self._curr_idx = 0
        return self

    def __len__(self) -> Optional[int]:
        return 1

    def __next__(self) -> TrainVal:
        """Define how to get next object.

        Returns:
            None, train dataset, validation dataset.

        """
        if self._curr_idx > 0:
            raise StopIteration

        self._curr_idx += 1

        sdf = cast(SparkDataFrame, self.train.data)
        sdf = sdf.withColumn(self.TRAIN_VAL_COLUMN, sf.lit(0))

        train_ds = cast(SparkDataset, self.train.empty())
        train_ds.set_data(sdf, self.train.features, self.train.roles, name=self.train.name)

        return train_ds, train_ds

    def freeze(self) -> 'SparkDummyIterator':
        return SparkDummyIterator(self.train.freeze())

    def unpersist(self, skip_val: bool = False):
        if not skip_val:
            self.train.unpersist()

    def get_validation_data(self) -> SparkDataset:
        return self.train

    def convert_to_holdout_iterator(self) -> "SparkHoldoutIterator":
        sds = cast(SparkDataset, self.train)
        assert sds.folds_column is not None, "Cannot convert to Holdout iterator when folds_column is not defined"
        return SparkHoldoutIterator(self.train, self.train)


class SparkHoldoutIterator(SparkBaseTrainValidIterator):
    """Simple one step iterator over one fold of SparkDataset"""

    def __init__(self, train: SparkDataset, valid: SparkDataset):
        super().__init__(train)
        self._valid = valid
        self._curr_idx = 0

    def __iter__(self) -> Iterable:
        self._curr_idx = 0
        return self

    def __len__(self) -> Optional[int]:
        return 1

    def __next__(self) -> TrainVal:
        """Define how to get next object.

        Returns:
            None, train dataset, validation dataset.

        """
        if self._curr_idx > 0:
            raise StopIteration

        # full_ds, train_part_ds, valid_part_ds = self._split_by_fold(self._curr_idx)
        self._curr_idx += 1

        return self.train, self._valid

    def freeze(self) -> 'SparkHoldoutIterator':
        return SparkHoldoutIterator(self.train.freeze(), self._valid.freeze())

    def unpersist(self, skip_val: bool = False):
        self.train.unpersist()
        if not skip_val:
            self._valid.unpersist()

    def get_validation_data(self) -> SparkDataset:
        # full_ds, train_part_ds, valid_part_ds = self._split_by_fold(fold=0)
        return self._valid

    def convert_to_holdout_iterator(self) -> "SparkHoldoutIterator":
        return self

    def apply_selector(self, selector: SparkSelectionPipeline) -> "SparkBaseTrainValidIterator":
        train_valid = super().apply_selector(selector)
        train_valid._valid = selector.select(train_valid._valid)
        return train_valid

    def apply_feature_pipeline(self, features_pipeline: SparkFeaturesPipeline) -> "SparkBaseTrainValidIterator":
        train_valid = super().apply_feature_pipeline(features_pipeline)
        train_valid._valid = features_pipeline.transform(train_valid._valid)
        return train_valid


class SparkFoldsIterator(SparkBaseTrainValidIterator):
    """Classic cv iterator.

    Folds should be defined in Reader, based on cross validation method.
    """

    def __init__(self, train: SparkDataset, n_folds: Optional[int] = None):
        """Creates iterator.

        Args:
            train: Dataset for folding.
            n_folds: Number of folds.

        """
        super().__init__(train)

        num_folds = train.data.select(sf.max(train.folds_column).alias("max")).first()["max"]
        self.n_folds = num_folds + 1
        if n_folds is not None:
            self.n_folds = min(self.n_folds, n_folds)

        self._base_train_frozen = train.frozen
        self._train_frozen = self._base_train_frozen
        self._val_frozen = self._base_train_frozen

    def __len__(self) -> int:
        """Get len of iterator.

        Returns:
            Number of folds.

        """
        return self.n_folds

    def __iter__(self) -> "SparkFoldsIterator":
        """Set counter to 0 and return self.

        Returns:
            Iterator for folds.

        """
        logger.debug("Creating folds iterator")

        self._curr_idx = 0

        return self

    def __next__(self) -> TrainVal:
        """Define how to get next object.

        Returns:
            None, train dataset, validation dataset.

        """
        logger.debug(f"The next valid fold num: {self._curr_idx}")

        if self._curr_idx == self.n_folds:
            logger.debug("No more folds to continue, stopping iterations")
            raise StopIteration

        full_ds, train_part_ds, valid_part_ds = self._split_by_fold(self._curr_idx)
        self._curr_idx += 1

        return train_part_ds, valid_part_ds

    def freeze(self) -> 'SparkFoldsIterator':
        return SparkFoldsIterator(self.train.freeze(), n_folds=self.n_folds)

    # @property
    # def train_frozen(self) -> bool:
    #     return self._train_frozen
    #
    # @property
    # def val_frozen(self) -> bool:
    #     return self._val_frozen
    #
    # @train_frozen.setter
    # def train_frozen(self, val: bool):
    #     self._base_train_frozen = val
    #     self.train.frozen = self._base_train_frozen or self._base_train_frozen or self._val_frozen
    #
    # @val_frozen.setter
    # def val_frozen(self, val: bool):
    #     self._val_frozen = val
    #     self.train.frozen = self._base_train_frozen or self._train_frozen or self._val_frozen

    def unpersist(self, skip_val: bool = False):
        if not skip_val:
            self.train.unpersist()

    def get_validation_data(self) -> SparkDataset:
        return self.train

    def convert_to_holdout_iterator(self) -> SparkHoldoutIterator:
        """Convert iterator to hold-out-iterator.

        Fold 0 is used for validation, everything else is used for training.

        Returns:
            new hold-out-iterator.

        """
        _, train, valid = self._split_by_fold(0)
        return SparkHoldoutIterator(train, valid)
