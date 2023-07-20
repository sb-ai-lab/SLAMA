"""Base class for selection pipelines."""
from abc import ABC
from copy import copy
from typing import List
from typing import Optional
from typing import cast

from lightautoml.dataset.base import LAMLDataset
from lightautoml.dataset.base import RolesDict
from lightautoml.pipelines.selection.base import ComposedSelector
from lightautoml.pipelines.selection.base import EmptySelector
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.pipelines.selection.base import SelectionPipeline
from lightautoml.validation.base import TrainValidIterator
from pandas import Series
from pyspark.ml import Transformer

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.pipelines.base import TransformerInputOutputRoles
from sparklightautoml.pipelines.features.base import SparkFeaturesPipeline
from sparklightautoml.utils import NoOpTransformer
from sparklightautoml.validation.base import SparkBaseTrainValidIterator
from sparklightautoml.validation.base import SparkSelectionPipeline


class SparkImportanceEstimator(ImportanceEstimator, ABC):
    def __init__(self):
        super(SparkImportanceEstimator, self).__init__()


class SparkSelectionPipelineWrapper(SparkSelectionPipeline, TransformerInputOutputRoles):
    def __init__(self, sel_pipe: SelectionPipeline):
        # assert not sel_pipe.is_fitted, "Cannot work with prefitted SelectionPipeline"
        self._validate_sel_pipe(sel_pipe)

        self._sel_pipe = sel_pipe
        self._service_columns = None
        self._is_fitted = False
        self._input_roles: Optional[RolesDict] = None
        self._output_roles: Optional[RolesDict] = None
        self._feature_pipeline = cast(SparkFeaturesPipeline, self._sel_pipe.features_pipeline)
        self._service_columns: Optional[List[str]] = None
        super().__init__()

    def _validate_sel_pipe(self, sel_pipe: SelectionPipeline):
        selectors = sel_pipe.selectors if isinstance(sel_pipe, ComposedSelector) else [sel_pipe]
        for selp in selectors:
            msg = (
                f"SelectionPipeline should either be EmptySelector or have SparkFeaturePipeline as "
                f"features_pipeline, but it is {type(selp)} and have {type(selp.features_pipeline)}"
            )
            assert (
                isinstance(selp, EmptySelector)
                or isinstance(selp, BugFixSelectionPipelineWrapper)
                or isinstance(selp.features_pipeline, SparkFeaturesPipeline)
            ), msg

    def _build_transformer(self, *args, **kwargs) -> Optional[Transformer]:
        if not self._sel_pipe.is_fitted:
            return None

        # we cannot select columns on infer time
        # because ml pipes are executed sequentially during predict
        # applying selector on predict may lead to loss of columns
        # required for the subsequent ml pipes
        return NoOpTransformer(name=f"{type(self)}")

    @property
    def input_roles(self) -> Optional[RolesDict]:
        return self._input_roles

    @property
    def output_roles(self) -> Optional[RolesDict]:
        return self._output_roles

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def selected_features(self) -> List[str]:
        return self._sel_pipe.selected_features

    @property
    def in_features(self) -> List[str]:
        return self._sel_pipe.in_features

    @property
    def dropped_features(self) -> List[str]:
        return self._sel_pipe.dropped_features

    def perform_selection(self, train_valid: Optional[TrainValidIterator]):
        self._sel_pipe.perform_selection(train_valid)

    def fit(self, train_valid: SparkBaseTrainValidIterator):
        self._service_columns = train_valid.train.service_columns

        if not self._sel_pipe.is_fitted:
            self._sel_pipe.fit(train_valid)
        self._is_fitted = True

        self._input_roles = copy(train_valid.train.roles)
        self._output_roles = {
            feat: role for feat, role in self._input_roles.items() if feat in self._sel_pipe.selected_features
        }
        self._service_columns = train_valid.train.service_columns

    def select(self, dataset: SparkDataset) -> SparkDataset:
        return cast(SparkDataset, self._sel_pipe.select(dataset))

    def map_raw_feature_importances(self, raw_importances: Series):
        return self._sel_pipe.map_raw_feature_importances(raw_importances)

    def get_features_score(self):
        return self._sel_pipe.get_features_score()

    def _get_service_columns(self) -> List[str]:
        return self._service_columns


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
