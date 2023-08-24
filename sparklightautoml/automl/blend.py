import logging

from abc import ABC
from copy import copy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import cast

import numpy as np

from lightautoml.automl.blend import WeightedBlender
from lightautoml.dataset.roles import ColumnRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.reader.base import RolesDict
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml import Transformer
from pyspark.ml.feature import SQLTransformer

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.ml_algo.base import AveragingTransformer
from sparklightautoml.pipelines.base import TransformerInputOutputRoles
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.tasks.base import DEFAULT_PREDICTION_COL_NAME
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.transformers.base import DropColumnsTransformer
from sparklightautoml.utils import ColumnsSelectorTransformer


logger = logging.getLogger(__name__)


class SparkBlender(TransformerInputOutputRoles, ABC):
    """Basic class for blending.

    Blender learns how to make blend
    on sequence of prediction datasets and prune pipes,
    that are not used in final blend.

    """

    def __init__(self):
        self._transformer = None
        self._single_prediction_col_name = DEFAULT_PREDICTION_COL_NAME
        self._pred_role: Optional[ColumnRole] = None
        self._input_roles: Optional[RolesDict] = None
        self._output_roles: Optional[RolesDict] = None
        self._task: Optional[SparkTask] = None
        self._service_columns: Optional[List[str]] = None
        super().__init__()

    @property
    def input_roles(self) -> Optional[RolesDict]:
        """Returns dict of input roles"""
        return self._input_roles

    @property
    def output_roles(self) -> Optional[RolesDict]:
        """Returns dict of output roles"""
        return self._output_roles

    def _get_service_columns(self) -> List[str]:
        return self._service_columns

    def transformer(self, *args, **kwargs) -> Optional[Transformer]:
        """Returns Spark MLlib Transformer.
        Represents a Transformer with fitted models."""

        assert self._transformer is not None, "Pipeline is not fitted!"

        return self._transformer

    def _build_transformer(self, *args, **kwargs) -> Optional[Transformer]:
        return self._transformer

    def fit_predict(
        self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]
    ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:
        logger.info(f"Blender {type(self)} starting fit_predict")

        self._set_metadata(predictions, pipes)

        if len(pipes) == 1 and len(pipes[0].ml_algos) == 1:
            statement = (
                f"SELECT *, {pipes[0].ml_algos[0].prediction_feature} "
                f"AS {self._single_prediction_col_name} FROM __THIS__"
            )

            logger.info(f"Select prediction columns with query: {statement}")

            self._transformer = Pipeline(
                stages=[
                    SQLTransformer(statement=statement),
                    DropColumnsTransformer(remove_cols=[pipes[0].ml_algos[0].prediction_feature]),
                ]
            ).fit(predictions.data)

            preds = predictions.empty()
            preds.set_data(
                self._transformer.transform(predictions.data),
                list(self.output_roles.keys()),
                self.output_roles,
                name=type(self).__name__,
            )

            return preds, pipes

        result = self._fit_predict(predictions, pipes)

        self._output_roles = result[0].roles
        self._service_columns = predictions.service_columns

        logger.info(f"Blender {type(self)} finished fit_predict")

        return result

    def predict(self, predictions: SparkDataset) -> SparkDataset:
        return self._make_transformed_dataset(predictions)

    def _fit_predict(
        self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]
    ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:
        raise NotImplementedError()

    # noinspection PyMethodMayBeStatic
    def split_models(self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]) -> List[Tuple[str, int, int]]:
        """Split predictions by single model prediction datasets.

        Args:
            predictions: Dataset with predictions.
            pipes: ml pipelines to be associated with the predictions

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
        assert len(predictions.roles) > 0, "The predictions dataset should have at least one column"
        self._input_roles = copy(predictions.roles)
        self._pred_role = list(predictions.roles.values())[0]
        self._output_roles = {self._single_prediction_col_name: self._pred_role}

        if isinstance(self._pred_role, NumericVectorOrArrayRole):
            self._outp_dim = self._pred_role.size
        else:
            self._outp_dim = 1
        self._outp_prob = predictions.task.name in ["binary", "multiclass"]
        self._score = predictions.task.get_dataset_metric()
        self._task = predictions.task

    def score(self, dataset: SparkDataset) -> float:
        """Score metric for blender.

        Args:
            dataset: Blended predictions dataset.

        Returns:
            Metric value.

        """
        return self._score(dataset, True)


class SparkBestModelSelector(SparkBlender, WeightedBlender):
    """Select best single model from level.

    Drops pipes that are not used in calc best model.
    Works in general case (even on some custom things)
    and most efficient on inference.
    Perform worse than other on tables,
    specially if some of models was terminated by timer.

    """

    def _fit_predict(
        self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]
    ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:
        """Simple fit - just take one best.

        Args:
            predictions: Sequence of datasets with predictions.
            pipes: Sequence of pipelines.

        Returns:
            Single prediction dataset and Sequence of pruned pipelines.

        """
        splitted_models_and_pipes = self.split_models(predictions, pipes)

        best_pred = None
        best_pred_col = None
        best_pipe_idx = 0
        best_model_idx = 0
        best_score = -np.inf

        for pred_col, mod, pipe in splitted_models_and_pipes:
            # pred_ds = self._make_single_pred_ds(predictions, pred_col)
            pred_ds = cast(SparkDataset, predictions[:, [pred_col]])
            score = self.score(pred_ds)

            if score > best_score:
                best_pipe_idx = pipe
                best_model_idx = mod
                best_score = score
                best_pred = pred_ds
                best_pred_col = pred_col

        best_pipe = pipes[best_pipe_idx]
        best_pipe.ml_algos = [best_pipe.ml_algos[best_model_idx]]

        self._transformer = Pipeline(
            stages=[
                SQLTransformer(
                    statement=f"SELECT *, {best_pred_col} AS {self._single_prediction_col_name} FROM __THIS__"
                ),
                ColumnsSelectorTransformer(
                    name=f"{type(self)}",
                    input_cols=[self._single_prediction_col_name],
                    optional_cols=predictions.service_columns,
                ),
            ]
        ).fit(best_pred.data)

        self._output_roles = {self._single_prediction_col_name: best_pred.roles[best_pred_col]}

        out_ds = best_pred.empty()
        out_ds.set_data(
            self._transformer.transform(best_pred.data),
            list(self._output_roles.keys()),
            self._output_roles,
            name=type(self).__name__,
        )

        return out_ds, [best_pipe]


class SparkWeightedBlender(SparkBlender, WeightedBlender):
    """Weighted Blender based on coord descent, optimize task metric directly.

    Weight sum eq. 1.
    Good blender for tabular data,
    even if some predictions are NaN (ex. timeout).
    Model with low weights will be pruned.

    """

    def __init__(
        self,
        max_iters: int = 5,
        max_inner_iters: int = 7,
        max_nonzero_coef: float = 0.05,
    ):
        SparkBlender.__init__(self)
        WeightedBlender.__init__(self, max_iters, max_inner_iters, max_nonzero_coef)
        self._predictions_dataset: Optional[SparkDataset] = None

    def _get_weighted_pred(
        self,
        splitted_preds: Sequence[str],
        wts: Optional[np.ndarray],
        remove_splitted_preds_cols: Optional[List[str]] = None,
    ) -> SparkDataset:
        avr = self._build_avr_transformer(splitted_preds, wts, remove_splitted_preds_cols)
        vsh = self._build_vector_size_hint(self._single_prediction_col_name, self._pred_role)

        weighted_preds_sdf = PipelineModel(stages=[avr, vsh]).transform(self._predictions_dataset.data)

        wpreds_sds = self._predictions_dataset.empty()
        wpreds_sds.set_data(
            weighted_preds_sdf, list(self.output_roles.keys()), self.output_roles, name=type(self).__name__
        )

        return wpreds_sds

    def _build_avr_transformer(
        self,
        splitted_preds: Sequence[str],
        wts: Optional[np.ndarray],
        remove_splitted_preds_cols: Optional[List[str]] = None,
    ) -> AveragingTransformer:
        remove_cols = list(splitted_preds)

        if remove_splitted_preds_cols is not None:
            remove_cols.extend(remove_splitted_preds_cols)

        return AveragingTransformer(
            task_name=self._task.name,
            input_cols=list(splitted_preds),
            output_col=self._single_prediction_col_name,
            remove_cols=remove_cols,
            convert_to_array_first=True,
            weights=(wts * len(wts)).tolist(),
            dim_num=self._outp_dim,
        )

    def _fit_predict(
        self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]
    ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:
        self._predictions_dataset = predictions

        sm = self.split_models(predictions, pipes)
        pred_cols = [pred_col for pred_col, _, _ in sm]
        pipe_idx = np.array([pidx for _, _, pidx in sm])

        wts = self._optimize(pred_cols)

        reweighted_pred_cols = [x for (x, w) in zip(pred_cols, wts) if w > 0]
        removed_cols = [x for x in pred_cols if x not in reweighted_pred_cols]
        _, self.wts = self._prune_pipe(pipes, wts, pipe_idx)
        pipes = cast(Sequence[SparkMLPipeline], pipes)

        self._transformer = self._build_avr_transformer(
            reweighted_pred_cols, self.wts, remove_splitted_preds_cols=removed_cols
        )
        outp = self._get_weighted_pred(reweighted_pred_cols, self.wts, remove_splitted_preds_cols=removed_cols)

        return outp, pipes


class SparkMeanBlender(SparkBlender):
    """Simple average level predictions.

    Works only with TabularDatasets.
    Doesn't require target to fit.
    No pruning.

    """

    def _fit_predict(
        self, predictions: SparkDataset, pipes: Sequence[SparkMLPipeline]
    ) -> Tuple[SparkDataset, Sequence[SparkMLPipeline]]:
        pred_cols = [pred_col for pred_col, _, _ in self.split_models(predictions, pipes)]

        avr = AveragingTransformer(
            task_name=predictions.task.name,
            input_cols=pred_cols,
            output_col=self._single_prediction_col_name,
            remove_cols=pred_cols,
            convert_to_array_first=not (predictions.task.name == "reg"),
            dim_num=self._outp_dim,
        )
        vsh = self._build_vector_size_hint(self._single_prediction_col_name, self._pred_role)

        self._transformer = PipelineModel(stages=[avr, vsh])

        df = self._transformer.transform(predictions.data)

        if predictions.task.name in ["binary", "multiclass"]:
            assert isinstance(self._pred_role, NumericVectorOrArrayRole)
            output_role = NumericVectorOrArrayRole(
                self._pred_role.size,
                "MeanBlend_{{}}",
                dtype=np.float32,
                prob=self._outp_prob,
                is_vector=self._pred_role.is_vector,
            )
        else:
            output_role = NumericRole(np.float32, prob=self._outp_prob)

        roles = {self._single_prediction_col_name: output_role}
        pred_ds = predictions.empty()
        pred_ds.set_data(df, list(roles.keys()), roles, name=type(self).__name__)

        self._output_roles = copy(roles)

        return pred_ds, pipes
