from typing import cast

from lightautoml.dataset.base import LAMLDataset
from lightautoml.spark.dataset import SparkDataset
from lightautoml.transformers.base import LAMLTransformer


class SparkTransformer(LAMLTransformer):
    def fit(self, dataset: LAMLDataset) -> "SparkTransformer":
        return cast(SparkTransformer, super().fit(dataset))

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        return cast(SparkDataset, super(SparkTransformer, self).transform(dataset))
