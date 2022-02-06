import logging
from typing import Optional, cast, Tuple

from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.validation.base import TrainValidIterator, HoldoutIterator

from pyspark.sql import functions as F


logger = logging.getLogger(__name__)


class FoldsIterator(TrainValidIterator):
    """Classic cv iterator.

    Folds should be defined in Reader, based on cross validation method.
    """

    def __init__(self, train: SparkDataset, n_folds: Optional[int] = None):
        """Creates iterator.

        Args:
            train: Dataset for folding.
            n_folds: Number of folds.

        """
        assert hasattr(train, "folds"), "Folds in dataset should be defined to make folds iterator."

        super().__init__(train)
        folds = cast(SparkDataFrame, train.folds)
        num_folds = folds.select(F.max(train.folds_column).alias("max")).first()["max"]
        self.n_folds = num_folds + 1
        if n_folds is not None:
            self.n_folds = min(self.n_folds, n_folds)

        self._df: Optional[SparkDataFrame] = None

    def __len__(self) -> int:
        """Get len of iterator.

        Returns:
            Number of folds.

        """
        return self.n_folds

    def __iter__(self) -> "FoldsIterator":
        """Set counter to 0 and return self.

        Returns:
            Iterator for folds.

        """
        logger.debug("Creating folds iterator")

        self._curr_idx = 0

        dataset = cast(SparkDataset, self.train)

        self._df = dataset.data.join(dataset.folds, SparkDataset.ID_COLUMN).cache()

        return self

    def __next__(self) -> Tuple[None, SparkDataset, SparkDataset]:
        """Define how to get next object.

        Returns:
            None, train dataset, validation dataset.

        """
        logger.debug(f"The next valid fold num: {self._curr_idx}")

        if self._curr_idx == self.n_folds:
            logger.debug("No more folds to continue, stopping iterations")
            self._df.unpersist()
            raise StopIteration

        train_ds, valid_ds = self.__split_by_fold(self._df, self._curr_idx)
        self._curr_idx += 1

        return None, train_ds, valid_ds

    def get_validation_data(self) -> SparkDataset:
        """Just return train dataset.

        Returns:
            Whole train dataset.

        """
        return self.train

    def convert_to_holdout_iterator(self) -> HoldoutIterator:
        """Convert iterator to hold-out-iterator.

        Fold 0 is used for validation, everything else is used for training.

        Returns:
            new hold-out-iterator.

        """
        # TODO: SPARK-LAMA need to uncache later
        dataset = cast(SparkDataset, self.train)
        df = dataset.data.join(dataset.folds, SparkDataset.ID_COLUMN).cache()

        train_ds, valid_ds = self.__split_by_fold(df, 0)

        return HoldoutIterator(train_ds, valid_ds)

    def __split_by_fold(self, df: SparkDataFrame, fold: int) -> Tuple[SparkDataset, SparkDataset]:
        train_df = df.where(F.col(self.train.folds_column) != fold).drop(self.train.folds_column)
        valid_df = df.where(F.col(self.train.folds_column) == fold).drop(self.train.folds_column)

        train_ds = cast(SparkDataset, self.train.empty())
        train_ds.set_data(train_df, self.train.features, self.train.roles)
        valid_ds = cast(SparkDataset, self.train.empty())
        valid_ds.set_data(valid_df, self.train.features, self.train.roles)

        return train_ds, valid_ds