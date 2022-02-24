from abc import ABC
from copy import copy
from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
from pyspark.ml import Transformer
from pyspark.ml.param import Params
from pyspark.ml.param.shared import HasInputCols, HasOutputCol, Param
from pyspark.ml.util import MLWritable
from pyspark.sql import functions as F
from pyspark.sql.functions import isnan

from lightautoml.automl.blend import Blender, \
    WeightedBlender as LAMAWeightedBlender
from lightautoml.dataset.roles import ColumnRole, NumericRole
from lightautoml.reader.base import RolesDict
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.ml_algo.base import AveragingTransformer
from lightautoml.spark.pipelines.ml.base import SparkMLPipeline
from lightautoml.spark.tasks.base import DEFAULT_PREDICTION_COL_NAME
from lightautoml.spark.transformers.base import ColumnsSelectorTransformer
from lightautoml.spark.utils import NoOpTransformer


class SparkBlender(ABC):
    """Basic class for blending.

    Blender learns how to make blend
    on sequence of prediction datasets and prune pipes,
    that are not used in final blend.

    """

    def __init__(self):
        super().__init__()
        self._transformer = None
        self._single_prediction_col_name = DEFAULT_PREDICTION_COL_NAME
        self._pred_role: Optional[ColumnRole] = None
        self._output_roles: Optional[RolesDict] = None

    @property
    def output_roles(self) -> RolesDict:
        assert self._output_roles is not None, "Blender has not been fitted yet"
        return self._output_roles

    @property
    def transformer(self) -> Transformer:
        """Returns Spark MLlib Transformer.
        Represents a Transformer with fitted models."""

        assert self._transformer is not None, "Pipeline is not fitted!"

        return self._transformer

    def fit_predict(
        self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]
    ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:

        if len(pipes) == 1 and len(pipes[0].ml_algos) == 1:
            self._transformer = ColumnsSelectorTransformer(
                input_cols=[SparkDataset.ID_COLUMN] + list(pipes[0].output_roles.keys()),
                optional_cols=[predictions.target_column] if predictions.target_column else []
            )
            self._output_roles = copy(predictions.roles)
            return predictions, pipes

        self._set_metadata(predictions, pipes)

        return self._fit_predict(predictions, pipes)

    def _fit_predict(self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]) \
        -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:
        raise NotImplementedError()

    def split_models(self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]) \
            -> List[Tuple[str, int, int]]:
        """Split predictions by single model prediction datasets.

        Args:
            predictions: Dataset with predictions.

        Returns:
            Each tuple in the list is:
            - prediction column name
            - corresponding model index (in the pipe)
            - corresponding pipe index

        """
        return [
            (ml_algo.prediction_feature, j, i)
            for i, pipe in enumerate(pipes)
            for j, ml_algo in enumerate(pipe.ml_algos)
        ]

    def _set_metadata(self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]):
        self._pred_role = predictions.roles[pipes[0].ml_algos[0].prediction_feature]

        if isinstance(self._pred_role, NumericVectorOrArrayRole):
            self._outp_dim = self._pred_role.size
        else:
            self._outp_dim = 1
        self._outp_prob = predictions.task.name in ["binary", "multiclass"]
        self._score = predictions.task.get_dataset_metric()

    def _make_single_pred_ds(self, predictions: SparkDataset, pred_col: str) -> SparkDataset:
        pred_sdf = predictions.data.select(
            SparkDataset.ID_COLUMN,
            predictions.target_column,
            F.col(pred_col).alias(self._single_prediction_col_name)
        )
        pred_roles = {c: predictions.roles[c] for c in pred_sdf.columns}
        pred_ds = predictions.empty()
        pred_ds.set_data(pred_sdf, pred_sdf.columns, pred_roles)

        return pred_ds

    def score(self, dataset: SparkDataset) -> float:
        """Score metric for blender.

        Args:
            dataset: Blended predictions dataset.

        Returns:
            Metric value.

        """
        return self._score(dataset, True)


class SparkBestModelSelector(SparkBlender):
    def _fit_predict(self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]) \
            -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:
        """Simple fit - just take one best.

                Args:
                    predictions: Sequence of datasets with predictions.
                    pipes: Sequence of pipelines.

                Returns:
                    Single prediction dataset and Sequence of pruned pipelines.

                """
        splitted_models_and_pipes = self.split_models(predictions, pipes)

        best_pred = None
        best_pipe_idx = 0
        best_model_idx = 0
        best_score = -np.inf

        for pred_col, mod, pipe in splitted_models_and_pipes:
            pred_ds = self._make_single_pred_ds(predictions, pred_col)
            score = self.score(pred_ds)

            if score > best_score:
                best_pipe_idx = pipe
                best_model_idx = mod
                best_score = score
                best_pred = pred_ds

        best_pipe = pipes[best_pipe_idx]
        best_pipe.ml_algos = [best_pipe.ml_algos[best_model_idx]]

        self._transformer = ColumnsSelectorTransformer(
            input_cols=[SparkDataset.ID_COLUMN, self._single_prediction_col_name]
        )

        self._output_roles = copy(best_pred.roles)

        return best_pred, [best_pipe]


class SparkWeightedBlender(SparkBlender):
    def __init__(self, max_nonzero_coef: float = 0.05):
        super().__init__()
        self.wts = None
        self._max_nonzero_coef = max_nonzero_coef

    def _fit_predict(self,
                     predictions: SparkDataset,
                     pipes: Sequence[SparkMLPipeline]
                     ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:

        pred_cols = [pred_col for pred_col, _, _ in self.split_models(predictions, pipes)]

        self._transformer = WeightedBlenderTransformer(
            task_name=predictions.task.name,
            input_cols=pred_cols,
            output_col=self._single_prediction_col_name,
            remove_cols=pred_cols,
            wts=self.wts
        )

        df = self._transformer.transform(predictions.data)

        if predictions.task.name in ["binary", "multiclass"]:
            assert isinstance(self._pred_role, NumericVectorOrArrayRole)
            output_role = NumericVectorOrArrayRole(
                self._pred_role.size,
                f"WeightedBlend_{{}}",
                dtype=np.float32,
                prob=self._outp_prob,
                is_vector=self._pred_role.is_vector
            )
        else:
            output_role = NumericRole(np.float32, prob=self._outp_prob)

        roles = {f: predictions.roles[f] for f in predictions.features if f not in pred_cols}
        roles[self._single_prediction_col_name] = output_role
        pred_ds = predictions.empty()
        pred_ds.set_data(df, df.columns, roles)

        self._output_roles = copy(roles)

        return pred_ds, pipes

    def fit_predict(self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]) -> Tuple[
        SparkDataset, Sequence[SparkMLPipeline]]:

        pred_cols = [pred_col for pred_col, _, _ in self.split_models(predictions, pipes)]

        length = len(pred_cols)

        if self.wts is None:
            self.wts = np.array([1.0 / length for col in pred_cols])
        else:
            assert len(self.wts) == length, 'Number of prediction cols and number of col weights must be equal'
            self.wts = np.array([w for col, w in zip(pred_cols, self.wts)])

        return super().fit_predict(predictions, pipes)


class WeightedBlenderTransformer(Transformer, HasInputCols, HasOutputCol, MLWritable):
    taskName = Param(Params._dummy(), "taskName", "task name")
    removeCols = Param(Params._dummy(), "removeCols", "cols to remove")
    wts = Param(Params._dummy(), "wts", "weights")

    def __init__(self,
                 task_name: str,
                 input_cols: List[str],
                 output_col: str,
                 wts: Optional[np.ndarray] = None,
                 remove_cols: Optional[List[str]] = None):
        super().__init__()
        self.set(self.taskName, task_name)
        self.set(self.inputCols, input_cols)
        self.set(self.outputCol, output_col)
        if not remove_cols:
            remove_cols = []
        self.set(self.removeCols, remove_cols)
        self.set(self.wts, wts)

    def getRemoveCols(self) -> List[str]:
        return self.getOrDefault(self.removeCols)

    def getWts(self) -> List[str]:
        return self.getOrDefault(self.wts)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        wts = self.getWts()
        pred_cols = self.getInputCols()
        if self.getOrDefault(self.taskName) in ["binary", "multiclass"]:
            def sum_arrays(x):
                is_all_nth_elements_nan = sum(F.when(isnan(x[c]), 1).otherwise(0) for c in pred_cols) == len(pred_cols)
                sum_weights_where_nan = sum(F.when(isnan(x[c]), wts[c]).otherwise(0.0) for c in pred_cols)
                sum_weights_where_nonnan = sum(F.when(isnan(x[c]), 0.0).otherwise(wts[c]) for c in pred_cols)
                # sum of non-nan nth elements multiplied by normalized weights
                weighted_sum = sum(F.when(isnan(x[c]), 0).otherwise(x[c]*(wts[c]+wts[c]*sum_weights_where_nan/sum_weights_where_nonnan)) for c in pred_cols)
                return F.when(is_all_nth_elements_nan, float('nan')) \
                        .otherwise(weighted_sum)
            out_col = F.transform(F.arrays_zip(*pred_cols), sum_arrays).alias(self.getOutputCol())
        else:
            is_all_columns_nan = sum(F.when(isnan(F.col(c)), 1).otherwise(0) for c in pred_cols) == len(pred_cols)
            sum_weights_where_nan = sum(F.when(isnan(F.col(c)), wts[c]).otherwise(0.0) for c in pred_cols)
            sum_weights_where_nonnan = sum(F.when(isnan(F.col(c)), 0.0).otherwise(wts[c]) for c in pred_cols)
            # sum of non-nan predictions multiplied by normalized weights
            weighted_sum = sum(F.when(isnan(F.col(c)), 0).otherwise(F.col(c)*(wts[c]+wts[c]*sum_weights_where_nan/sum_weights_where_nonnan)) for c in pred_cols)
            out_col = F.when(is_all_columns_nan, float('nan')).otherwise(weighted_sum).alias(self.getOutputCol())

        cols_to_remove = set(self.getRemoveCols())
        cols_to_select = [c for c in dataset.columns if c not in cols_to_remove]
        out_df = dataset.select(*cols_to_select, out_col)
        return out_df

    def write(self):
        pass


class SparkMeanBlender(SparkBlender):
    def _fit_predict(self,
                     predictions: SparkDataset,
                     pipes: Sequence[SparkMLPipeline]
                     ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:

        pred_cols = [pred_col for pred_col, _, _ in self.split_models(predictions, pipes)]

        self._transformer = AveragingTransformer(
            task_name=predictions.task.name,
            input_cols=pred_cols,
            output_col=self._single_prediction_col_name,
            remove_cols=pred_cols,
            convert_to_array_first=not (predictions.task.name == "reg"),
            dim_num=self._outp_dim
        )

        df = self._transformer.transform(predictions.data)

        if predictions.task.name in ["binary", "multiclass"]:
            assert isinstance(self._pred_role, NumericVectorOrArrayRole)
            output_role = NumericVectorOrArrayRole(
                self._pred_role.size,
                f"MeanBlend_{{}}",
                dtype=np.float32,
                prob=self._outp_prob,
                is_vector=self._pred_role.is_vector
            )
        else:
            output_role = NumericRole(np.float32, prob=self._outp_prob)

        roles = {f: predictions.roles[f] for f in predictions.features if f not in pred_cols}
        roles[self._single_prediction_col_name] = output_role
        pred_ds = predictions.empty()
        pred_ds.set_data(df, df.columns, roles)

        self._output_roles = copy(roles)

        return pred_ds, pipes

