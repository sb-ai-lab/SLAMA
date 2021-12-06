from collections import Sequence
from typing import Union, Tuple, Optional

from lightautoml.ml_algo.tuning.base import ParamsTuner
from lightautoml.pipelines.ml.nested_ml_pipe import NestedTabularMLPipeline as LAMANestedTabularMLPipeline
from lightautoml.pipelines.selection.base import SelectionPipeline
from lightautoml.spark.ml_algo.base import TabularMLAlgo
from lightautoml.spark.pipelines.features.base import FeaturesPipeline
from lightautoml.spark.pipelines.ml.base import SparkMLPipelineMixin


class NestedTabularMLPipeline(SparkMLPipelineMixin, LAMANestedTabularMLPipeline):
    """
        Same as NestedTabularMLPipeline of LAMA, but redefines a couple of methods via SparkMLPipelineMixin
    """

    def __init__(
            self,
            ml_algos: Sequence[Union[TabularMLAlgo, Tuple[TabularMLAlgo, ParamsTuner]]],
            force_calc: Union[bool, Sequence[bool]] = True,
            pre_selection: Optional[SelectionPipeline] = None,
            features_pipeline: Optional[FeaturesPipeline] = None,
            post_selection: Optional[SelectionPipeline] = None,
            cv: int = 1,
            n_folds: Optional[int] = None,
            inner_tune: bool = False,
            refit_tuner: bool = False,
    ):
        super().__init__(ml_algos, force_calc, pre_selection, features_pipeline,
                         post_selection, cv, n_folds, inner_tune, refit_tuner)
    pass
