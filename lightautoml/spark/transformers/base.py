from typing import cast

from lightautoml.dataset.base import LAMLDataset
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.transformers.base import LAMLTransformer


class SparkTransformer(LAMLTransformer):

    _features = []

    def fit(self, dataset: SparkDataset) -> "SparkTransformer":

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
        self.fit(dataset)
        return self.transform(dataset)
