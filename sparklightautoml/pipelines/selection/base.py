"""Base class for selection pipelines."""
from copy import copy
from typing import Any, Optional, List

from lightautoml.dataset.base import LAMLDataset
from lightautoml.pipelines.selection.base import SelectionPipeline
from lightautoml.validation.base import TrainValidIterator
from pandas import Series

from sparklightautoml.dataset.base import SparkDataset


class SparkImportanceEstimator:
    """
    Abstract class, that estimates feature importances.
    """

    def __init__(self):
        self.raw_importances = None

    # Change signature here to be compatible with MLAlgo
    def fit(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def get_features_score(self) -> SparkDataset:

        return self.raw_importances


class BugFixSelectionPipelineWrapper(SelectionPipeline):
    def __init__(self, instance: SelectionPipeline):
        super(BugFixSelectionPipelineWrapper, self).__init__()
        self._instance = instance

    def fit(self, train_valid: TrainValidIterator):
        self._instance.fit(train_valid)

    def map_raw_feature_importances(self, raw_importances: Series):
        return self._instance.map_raw_feature_importances(raw_importances)

    def get_features_score(self):
        return self._instance.get_features_score()

    @property
    def is_fitted(self) -> bool:
        return self._instance.is_fitted

    @property
    def selected_features(self) -> List[str]:
        return self._instance.selected_features

    @property
    def in_features(self) -> List[str]:
        return self._instance.in_features

    @property
    def dropped_features(self) -> List[str]:
        return self._instance.dropped_features

    def perform_selection(self, train_valid: Optional[TrainValidIterator]):
        self._instance.perform_selection(train_valid)

    def select(self, dataset: LAMLDataset) -> LAMLDataset:
        selected_features = copy(self._instance.selected_features)
        # Add features that forces input
        sl_set = set(selected_features)
        roles = dataset.roles
        for col in (x for x in dataset.features if x not in sl_set):
            if roles[col].force_input:
                if col not in sl_set:
                    selected_features.append(col)

        return dataset[:, selected_features]
