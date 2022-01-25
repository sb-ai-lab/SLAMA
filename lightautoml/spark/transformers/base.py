import logging
from typing import cast, Sequence, List, Set

from lightautoml.dataset.utils import concatenate
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.utils import log_exec_time
from lightautoml.transformers.base import LAMLTransformer, ColumnsSelector as LAMAColumnsSelector, \
    ChangeRoles as LAMAChangeRoles


logger = logging.getLogger(__name__)


class SparkTransformer(LAMLTransformer):

    _features = []

    _can_unwind_parents: bool = True

    def fit(self, dataset: SparkDataset) -> "SparkTransformer":

        logger.info(f"SparkTransformer of type: {type(self)}")
        self._features = dataset.features
        for check_func in self._fit_checks:
            check_func(dataset)

        return self._fit(dataset)

    def _fit(self, dataset: SparkDataset) -> "SparkTransformer":
        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        for check_func in self._transform_checks:
            check_func(dataset)

        return self._transform(dataset)

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        return dataset

    def fit_transform(self, dataset: SparkDataset) -> SparkDataset:
        # TODO: SPARK-LAMA probably we should assume
        #  that fit_transform executes with cache by default
        #  e.g fit_transform returns a cached and materialized dataset
        logger.info(f"fit_transform in {self._fname_prefix}: {type(self)}")

        dataset.cache()
        self.fit(dataset)

        # when True, it means that during fit operation we conducted some action that
        # materialized our current dataset and thus we can unpersist all its dependencies
        # because we have data to propagate in the cache already
        if self._can_unwind_parents:
            dataset.unwind_dependencies()
            deps = [dataset]
        else:
            deps = dataset.dependencies

        result = self.transform(dataset)
        result.dependencies = deps

        return result


class SequentialTransformer(SparkTransformer):
    """
    Transformer that contains the list of transformers and apply one by one sequentially.
    """
    _fname_prefix = "seq"

    def __init__(self, transformer_list: Sequence[SparkTransformer], is_already_fitted: bool = False):
        """

        Args:
            transformer_list: Sequence of transformers.

        """
        self.transformer_list = transformer_list
        self._is_fitted = is_already_fitted

    def _fit(self, dataset: SparkDataset) -> "SparkTransformer":
        """Fit not supported. Needs output to fit next transformer.

        Args:
            dataset: Dataset to fit.

        """
        raise NotImplementedError("Sequential supports only fit_transform.")

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        """Apply the sequence of transformers to dataset one over output of previous.

        Args:
            dataset: Dataset to transform.

        Returns:
            Dataset with new features.

        """
        logger.info(f"[{type(self)} (SEQ)] transform is started")
        for trf in self.transformer_list:
            dataset = trf.transform(dataset)

        logger.info(f"[{type(self)} (SEQ)] transform is finished")

        return dataset

    def fit_transform(self, dataset: SparkDataset) -> SparkDataset:
        """Sequential ``.fit_transform``.

         Output features - features from last transformer with no prefix.

        Args:
            dataset: Dataset to transform.

        Returns:
            Dataset with new features.

        """
        logger.info(f"[{type(self)} (SEQ)] fit_transform is started")
        if not self._is_fitted:
            for trf in self.transformer_list:
                dataset = trf.fit_transform(dataset)
        else:
            dataset = self.transform(dataset)

        self.features = self.transformer_list[-1].features
        logger.info(f"[{type(self)} (SEQ)] fit_transform is finished")
        return dataset


class UnionTransformer(SparkTransformer):
    """Transformer that apply the sequence on transformers in parallel on dataset and concatenate the result."""

    _fname_prefix = "union"

    def __init__(self, transformer_list: Sequence[SparkTransformer], n_jobs: int = 1):
        """

        Args:
            transformer_list: Sequence of transformers.
            n_jobs: Number of processes to run fit and transform.

        """
        # TODO: Add multiprocessing version here
        self.transformer_list = [x for x in transformer_list if x is not None]
        self.n_jobs = n_jobs

        assert len(self.transformer_list) > 0, "The list of transformers cannot be empty or contains only None-s"

    def _fit(self, dataset: SparkDataset) -> "SparkTransformer":
        assert self.n_jobs == 1, f"Number of parallel jobs is now limited to only 1"

        fnames = []
        logger.info(f"[{type(self)} (UNI)] fit is started")
        with dataset.applying_temporary_caching():
            for trf in self.transformer_list:
                trf.fit(dataset)
                fnames.append(trf.features)

        self.features = fnames
        logger.info(f"[{type(self)} (UNI)] fit is finished")
        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        assert self.n_jobs == 1, f"Number of parallel jobs is now limited to only 1"

        res = []
        logger.info(f"[{type(self)} (UNI)] transform is started")
        for trf in self.transformer_list:
            ds = trf.transform(dataset)
            res.append(ds)

        union_res = SparkDataset.concatenate(res)
        logger.info(f"[{type(self)} (UNI)] transform is finished")
        return union_res

    def fit_transform(self, dataset: SparkDataset) -> SparkDataset:
        """Fit and transform transformers in parallel.
         Output names - concatenation of features names with no prefix.

        Args:
            dataset: Dataset to fit and transform on.

        Returns:
            Dataset with new features.

        """
        res = []
        actual_transformers = []
        logger.info(f"[{type(self)} (UNI)] fit_transform is started")

        with dataset.applying_temporary_caching():
            for trf in self.transformer_list:
                ds = trf.fit_transform(dataset)
                # if ds:
                res.append(ds)
                actual_transformers.append(trf)

        # this concatenate operations also propagates all dependencies
        logger.info(f"[{type(self)} (UNI)] fit_transform: concat is started")
        result = SparkDataset.concatenate(res) if len(res) > 0 else None
        logger.info(f"[{type(self)} (UNI)] fit_transform: concat is finished")

        self.transformer_list = actual_transformers
        self.features = result.features
        logger.info(f"[{type(self)} (UNI)] fit_transform is finished")
        return result


class ColumnsSelector(LAMAColumnsSelector, SparkTransformer):
    _fname_prefix = "colsel"
    _can_unwind_parents = False


class ChangeRoles(LAMAChangeRoles, SparkTransformer):
    _fname_prefix = "changeroles"
    _can_unwind_parents = False

