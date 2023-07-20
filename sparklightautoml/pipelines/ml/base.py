"""Base classes for MLPipeline."""
import uuid
import warnings

from copy import copy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import cast

from lightautoml.dataset.base import RolesDict
from lightautoml.ml_algo.tuning.base import ParamsTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from lightautoml.pipelines.ml.base import MLPipeline as LAMAMLPipeline
from lightautoml.pipelines.selection.base import EmptySelector
from pyspark.ml import PipelineModel
from pyspark.ml import Transformer

from ...computations.base import ComputationsSettings
from ...computations.builder import build_computations_manager
from ...dataset.base import SparkDataset
from ...ml_algo.base import SparkTabularMLAlgo
from ...validation.base import SparkBaseTrainValidIterator
from ..base import TransformerInputOutputRoles
from ..features.base import SparkEmptyFeaturePipeline
from ..features.base import SparkFeaturesPipeline
from ..selection.base import SparkSelectionPipelineWrapper


class SparkMLPipeline(LAMAMLPipeline, TransformerInputOutputRoles):
    """Spark version of :class:`~lightautoml.pipelines.ml.base.MLPipeline`. Single ML pipeline.

    Merge together stage of building ML model
    (every step, excluding model training, is optional):

        - Pre selection: select features from input data.
          Performed by
          :class:`~lightautoml.pipelines.selection.base.SelectionPipeline`.
        - Features generation: build new features from selected.
          Performed by
          :class:`~sparklightautoml.pipelines.features.base.SparkFeaturesPipeline`.
        - Post selection: One more selection step - from created features.
          Performed by
          :class:`~lightautoml.pipelines.selection.base.SelectionPipeline`.
        - Hyperparams optimization for one or multiple ML models.
          Performed by
          :class:`~lightautoml.ml_algo.tuning.base.ParamsTuner`.
        - Train one or multiple ML models:
          Performed by :class:`~sparklightautoml.ml_algo.base.SparkTabularMLAlgo`.
          This step is the only required for at least 1 model.

    """

    def __init__(
        self,
        ml_algos: Sequence[Union[SparkTabularMLAlgo, Tuple[SparkTabularMLAlgo, ParamsTuner]]],
        force_calc: Union[bool, Sequence[bool]] = True,
        pre_selection: Optional[SparkSelectionPipelineWrapper] = None,
        features_pipeline: Optional[SparkFeaturesPipeline] = None,
        post_selection: Optional[SparkSelectionPipelineWrapper] = None,
        name: Optional[str] = None,
        persist_before_ml_algo: bool = False,
        computations_settings: Optional[ComputationsSettings] = None,
    ):
        if features_pipeline is None:
            features_pipeline = SparkEmptyFeaturePipeline()

        if pre_selection is None:
            pre_selection = SparkSelectionPipelineWrapper(EmptySelector())

        if post_selection is None:
            post_selection = SparkSelectionPipelineWrapper(EmptySelector())

        super().__init__(ml_algos, force_calc, pre_selection, features_pipeline, post_selection)

        self._output_features = None
        self._output_roles = None
        self._transformer: Optional[Transformer] = None
        self._name = name if name else str(uuid.uuid4())[:5]
        self.ml_algos: List[SparkTabularMLAlgo] = []
        self.pre_selection = cast(SparkSelectionPipelineWrapper, self.pre_selection)
        self.post_selection = cast(SparkSelectionPipelineWrapper, self.post_selection)
        self.features_pipeline = cast(SparkFeaturesPipeline, self.features_pipeline)
        self._milestone_name = f"MLPipe_{self._name}"
        self._input_roles: Optional[RolesDict] = None
        self._output_roles: Optional[RolesDict] = None
        self._persist_before_ml_algo = persist_before_ml_algo
        self._service_columns: Optional[List[str]] = None
        self._computations_manager = build_computations_manager(computations_settings)

    @property
    def input_roles(self) -> Optional[RolesDict]:
        return self._input_roles

    @property
    def output_roles(self) -> Optional[RolesDict]:
        return self._output_roles

    @property
    def name(self) -> str:
        return self._name

    def _build_transformer(self, *args, **kwargs) -> Optional[Transformer]:
        assert self._transformer is not None, f"{type(self)} seems to be not fitted"
        return self._transformer

    def fit_predict(self, train_valid: SparkBaseTrainValidIterator) -> SparkDataset:
        """Fit on train/valid iterator and transform on validation part.

        Args:
            train_valid: Dataset iterator.

        Returns:
            Dataset with predictions of all models.

        """

        # train and apply pre selection
        train_valid = train_valid.apply_selector(self.pre_selection)

        # apply features pipeline
        train_valid = train_valid.apply_feature_pipeline(self.features_pipeline)

        # train and apply post selection
        train_valid = train_valid.apply_selector(self.post_selection)

        with train_valid.frozen() as frozen_train_valid:

            def build_fit_func(ml_algo: SparkTabularMLAlgo, param_tuner: ParamsTuner, force_calc: bool):
                def func():
                    fitted_ml_algo, curr_preds = tune_and_fit_predict(
                        ml_algo, param_tuner, frozen_train_valid, force_calc
                    )
                    fitted_ml_algo = cast(SparkTabularMLAlgo, fitted_ml_algo)
                    curr_preds = cast(SparkDataset, curr_preds)

                    if ml_algo is None:
                        warnings.warn(
                            "Current ml_algo has not been trained by some reason. " "Check logs for more details.",
                            RuntimeWarning,
                        )

                    return fitted_ml_algo, curr_preds

                return func

            fit_tasks = [
                build_fit_func(ml_algo, param_tuner, force_calc)
                for ml_algo, param_tuner, force_calc in zip(self._ml_algos, self.params_tuners, self.force_calc)
            ]

            results = self._computations_manager.compute(fit_tasks)
            ml_algos, preds = [list(el) for el in zip(*results)]
            self.ml_algos.extend(ml_algos)

            assert (
                len(self.ml_algos) > 0
            ), "Pipeline finished with 0 models for some reason.\nProbably one or more models failed"

        del self._ml_algos

        val_preds_ds = SparkDataset.concatenate(
            preds, name=f"{type(self)}_folds_predictions", extra_dependencies=[train_valid]
        )

        self._transformer = PipelineModel(
            stages=[
                # self.pre_selection.transformer(),
                self.features_pipeline.transformer(),
                # self.post_selection.transformer(),
                *[ml_algo.transformer() for ml_algo in self.ml_algos],
            ]
        )

        self._input_roles = copy(train_valid.train.roles)
        self._output_roles = copy(val_preds_ds.roles)
        self._service_columns = train_valid.train.service_columns

        return val_preds_ds

    def predict(self, dataset: SparkDataset) -> SparkDataset:
        """Predict on new dataset.

        Args:
            dataset: Dataset used for prediction.

        Returns:
            Dataset with predictions of all trained models.

        """
        return self._make_transformed_dataset(dataset)

    def _get_service_columns(self) -> List[str]:
        return self._service_columns
