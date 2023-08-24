import functools
import logging

from abc import ABC
from copy import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import cast

from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.base import MLAlgo
from lightautoml.utils.timer import TaskTimer
from pyspark.ml import Model
from pyspark.ml import PipelineModel
from pyspark.ml import Transformer
from pyspark.ml.param import Params
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.param.shared import HasOutputCol
from pyspark.ml.param.shared import Param
from pyspark.ml.util import DefaultParamsReadable
from pyspark.ml.util import DefaultParamsWritable
from pyspark.sql import functions as sf

from sparklightautoml.computations.base import ComputationSlot
from sparklightautoml.computations.base import ComputationsManager
from sparklightautoml.computations.base import ComputationsSettings
from sparklightautoml.computations.builder import build_computations_manager
from sparklightautoml.dataset.base import PersistenceLevel
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.pipelines.base import TransformerInputOutputRoles
from sparklightautoml.spark_functions import scalar_averaging
from sparklightautoml.spark_functions import vector_averaging
from sparklightautoml.utils import SparkDataFrame
from sparklightautoml.utils import log_exception
from sparklightautoml.validation.base import SparkBaseTrainValidIterator


logger = logging.getLogger(__name__)

SparkMLModel = PipelineModel

ComputationalParameters = Union[Dict[str, Any], ComputationsManager]


class SparkTabularMLAlgo(MLAlgo, TransformerInputOutputRoles, ABC):
    """Machine learning algorithms that accepts numpy arrays as input."""

    _name: str = "SparkTabularMLAlgo"
    _default_validation_col_name: str = SparkBaseTrainValidIterator.TRAIN_VAL_COLUMN

    def __init__(
        self,
        default_params: Optional[dict] = None,
        freeze_defaults: bool = True,
        timer: Optional[TaskTimer] = None,
        optimization_search_space: Optional[dict] = None,
        persist_output_dataset: bool = True,
        computations_settings: Optional[ComputationsSettings] = None,
    ):
        optimization_search_space = optimization_search_space if optimization_search_space else dict()
        super().__init__(default_params, freeze_defaults, timer, optimization_search_space)
        self.n_classes: Optional[int] = None
        self.persist_output_dataset = persist_output_dataset
        # names of columns that should contain predictions of individual models
        self._models_prediction_columns: Optional[List[str]] = None

        self._prediction_role: Optional[Union[NumericRole, NumericVectorOrArrayRole]] = None
        self._input_roles: Optional[RolesDict] = None
        self._service_columns: Optional[List[str]] = None
        self._computations_manager: Optional[ComputationsManager] = build_computations_manager(computations_settings)

    @property
    def features(self) -> Optional[List[str]]:
        """Get list of features."""
        return list(self._input_roles.keys()) if self._input_roles else None

    @features.setter
    def features(self, val: Sequence[str]):
        """List of features."""
        raise NotImplementedError("Unsupported operation")

    @property
    def input_roles(self) -> Optional[RolesDict]:
        return self._input_roles

    @property
    def output_roles(self) -> Optional[RolesDict]:
        return {self.prediction_feature: self.prediction_role}

    @property
    def prediction_feature(self) -> str:
        # return self._prediction_col
        return f"{self._name}_prediction"

    @property
    def prediction_role(self) -> Union[NumericRole, NumericVectorOrArrayRole]:
        return self._prediction_role

    @property
    def validation_column(self) -> str:
        return self._default_validation_col_name

    @property
    def computations_manager(self) -> ComputationsManager:
        return self._computations_manager

    @computations_manager.setter
    def computations_manager(self, value: ComputationsManager):
        self._computations_manager = value

    @log_exception(logger=logger)
    def fit_predict(self, train_valid_iterator: SparkBaseTrainValidIterator) -> SparkDataset:
        """Fit and then predict accordig the strategy that uses train_valid_iterator.

        If item uses more then one time it will
        predict mean value of predictions.
        If the element is not used in training then
        the prediction will be ``numpy.nan`` for this item

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Dataset with predicted values.

        """
        logger.info(f"Input columns for MLALgo: {sorted(train_valid_iterator.train.features)}")
        logger.info(f"Train size for MLAlgo: {train_valid_iterator.train.data.count()}")

        assert not self.is_fitted, "Algo is already fitted"

        self._input_roles = copy(train_valid_iterator.train.roles)
        # init params on input if no params was set before
        if self._params is None:
            self.params = self.init_params_on_input(train_valid_iterator)

        iterator_len = len(train_valid_iterator)
        if iterator_len > 1:
            logger.info("Start fitting \x1b[1m{}\x1b[0m ...".format(self._name))
            logger.debug(f"Training params: {self.params}")

        # get metric and loss if None
        self.task = train_valid_iterator.train.task

        valid_ds = cast(SparkDataset, train_valid_iterator.get_validation_data())

        self._infer_and_set_prediction_role(valid_ds)

        self._models_prediction_columns = []

        with train_valid_iterator.frozen() as frozen_train_valid_iterator:
            self.models, preds_dfs, self._models_prediction_columns = self._parallel_fit(
                train_valid_iterator=frozen_train_valid_iterator
            )

        full_preds_df = self._combine_val_preds(train_valid_iterator.get_validation_data(), preds_dfs)
        full_preds_df = self._build_averaging_transformer().transform(full_preds_df)
        full_preds_df = self._build_vector_size_hint(self.prediction_feature, self._prediction_role).transform(
            full_preds_df
        )

        pred_ds = valid_ds.empty()
        pred_ds.set_data(
            full_preds_df,
            list(self.output_roles.keys()),
            self.output_roles,
            dependencies=[train_valid_iterator],
            name=f"{self._name}",
        )
        if self.persist_output_dataset:
            pred_ds = pred_ds.persist(level=PersistenceLevel.REGULAR)

        self._service_columns = train_valid_iterator.train.service_columns

        if iterator_len > 1:
            single_pred_ds = self._make_single_prediction_dataset(pred_ds)
            logger.info(
                f"Fitting \x1b[1m{self._name}\x1b[0m finished. score = \x1b[1m{self.score(single_pred_ds)}\x1b[0m"
            )

        if iterator_len > 1 or "Tuned" not in self._name:
            logger.info("\x1b[1m{}\x1b[0m fitting and predicting completed".format(self._name))

        return pred_ds

    def fit_predict_single_fold(
        self,
        fold_prediction_column: str,
        validation_column: str,
        train: SparkDataset,
        runtime_settings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            fold_prediction_column: column name for predictions made for this fold
            validation_column: name of the column that signals if this row is from train or val
            train: dataset containing both train and val rows.
            runtime_settings: settings important for parallelism and performance that can depend on running processes
            at the moment

        Returns:
            Target predictions for valid dataset.

        """
        raise NotImplementedError

    def predict_single_fold(self, model: SparkMLModel, dataset: SparkDataset) -> SparkDataFrame:
        raise NotImplementedError("Not supported for Spark. Use transformer property instead ")

    def predict(self, dataset: SparkDataset) -> SparkDataset:
        return self._make_transformed_dataset(dataset)

    def _get_service_columns(self) -> List[str]:
        return self._service_columns

    def _infer_and_set_prediction_role(self, valid_ds: SparkDataset):
        outp_dim = 1
        if self.task.name == "multiclass":
            outp_dim = valid_ds.data.select(sf.max(valid_ds.target_column).alias("max")).first()
            outp_dim = outp_dim["max"] + 1
            self._prediction_role = NumericVectorOrArrayRole(
                outp_dim, f"{self.prediction_feature}" + "_{}", force_input=True, prob=True
            )
        elif self.task.name == "binary":
            outp_dim = 2
            self._prediction_role = NumericVectorOrArrayRole(
                outp_dim, f"{self.prediction_feature}" + "_{}", force_input=True, prob=True
            )
        else:
            self._prediction_role = NumericRole(force_input=True)

        self.n_classes = outp_dim

    @staticmethod
    def _get_predict_column(model: SparkMLModel) -> str:
        try:
            return model.getPredictionCol()
        except AttributeError:
            if isinstance(model, PipelineModel):
                return model.stages[-1].getPredictionCol()

            raise TypeError("Unknown model type! Unable ro retrieve prediction column")

    def _predict_feature_name(self):
        return f"{self._name}_prediction"

    def _build_averaging_transformer(self) -> Transformer:
        raise NotImplementedError()

    def _make_single_prediction_dataset(self, dataset: SparkDataset) -> SparkDataset:
        preds = dataset.data.select(SparkDataset.ID_COLUMN, dataset.target_column, self.prediction_feature)
        roles = {self.prediction_feature: dataset.roles[self.prediction_feature]}

        output: SparkDataset = dataset.empty()
        output.set_data(preds, list(roles.keys()), roles, name="single_prediction_dataset")

        return output

    def _combine_val_preds(self, val_data: SparkDataset, val_preds: Sequence[SparkDataFrame]) -> SparkDataFrame:
        # depending on train_valid logic there may be several ways of treating predictions results:
        # 1. for folds iterators - join the results, it will yield the full train dataset
        # 2. for holdout iterators - create None predictions in train_part and join with valid part
        # 3. for custom iterators which may put the same records in
        #   different folds: union + groupby + (optionally) union with None-fied train_part
        # 4. for dummy - do nothing
        assert len(val_preds) > 0

        if len(val_preds) == 1:
            return val_preds[0]

        # we leave only service columns, e.g. id, fold, target columns
        initial_df = cast(SparkDataFrame, val_data[:, []].data)

        full_val_preds = functools.reduce(
            lambda acc, x: acc.join(x, on=SparkDataset.ID_COLUMN, how="left"),
            [val_pred.drop(val_data.target_column) for val_pred in val_preds],
            initial_df,
        )

        return full_val_preds

    def _parallel_fit(
        self, train_valid_iterator: SparkBaseTrainValidIterator
    ) -> Tuple[List[Model], List[SparkDataFrame], List[str]]:
        num_folds = len(train_valid_iterator)

        def _fit_and_val_on_fold(
            fold_id: int, slot: ComputationSlot
        ) -> Optional[Tuple[int, Model, SparkDataFrame, str]]:
            mdl_pred_col = f"{self.prediction_feature}_{fold_id}"
            if num_folds > 1:
                logger.info2(
                    "===== Start working with \x1b[1mfold {}\x1b[0m for \x1b[1m{}\x1b[0m "
                    "=====".format(fold_id, self._name)
                )

            if self.timer.time_limit_exceeded():
                logger.info(f"No time to calculate fold {fold_id}/{num_folds} (Time limit is already exceeded)")
                return None

            runtime_settings = {"num_tasks": slot.num_tasks, "num_threads": slot.num_threads_per_executor}
            train_ds = slot.dataset

            mdl, vpred, _ = self.fit_predict_single_fold(
                mdl_pred_col, self.validation_column, train_ds, runtime_settings
            )
            vpred = vpred.select(SparkDataset.ID_COLUMN, train_ds.target_column, mdl_pred_col)

            return fold_id, mdl, vpred, mdl_pred_col

        results = self.computations_manager.compute_folds(train_valid_iterator, _fit_and_val_on_fold)

        self.timer.write_run_info()

        computed_results = (r for r in results if r is not None)
        _, models, val_preds, model_prediction_cols = [list(el) for el in zip(*computed_results)]

        return models, val_preds, model_prediction_cols


class AveragingTransformer(Transformer, HasInputCols, HasOutputCol, DefaultParamsWritable, DefaultParamsReadable):
    """
    Transformer that gets one or more columns and produce column with average values.
    """

    taskName = Param(Params._dummy(), "taskName", "task name")
    removeCols = Param(Params._dummy(), "removeCols", "cols to remove")
    convertToArrayFirst = Param(Params._dummy(), "convertToArrayFirst", "convert to array first")
    weights = Param(Params._dummy(), "weights", "weights")
    dimNum = Param(Params._dummy(), "dimNum", "dim num")

    def __init__(
        self,
        task_name: str = None,
        input_cols: Optional[List[str]] = None,
        output_col: str = "averaged_values",
        remove_cols: Optional[List[str]] = None,
        convert_to_array_first: bool = False,
        weights: Optional[List[int]] = None,
        dim_num: int = 1,
    ):
        """
        Args:
            task_name (str, optional): Task name: "binary", "multiclass" or "reg".
            input_cols (List[str], optional): List of input columns.
            output_col (str, optional): Output column name. Defaults to "averaged_values".
            remove_cols (Optional[List[str]], optional): Columns need to remove. Defaults to None.
            convert_to_array_first (bool, optional): If `True` then will be convert input vectors to arrays.
                Defaults to False.
            weights (Optional[List[int]], optional): List of weights to scaling output values. Defaults to None.
            dim_num (int, optional): Dimension of input columns. Defaults to 1.
        """
        super().__init__()
        input_cols = input_cols if input_cols else []
        self.set(self.taskName, task_name)
        self.set(self.inputCols, input_cols)
        self.set(self.outputCol, output_col)
        if not remove_cols:
            remove_cols = []
        self.set(self.removeCols, remove_cols)
        self.set(self.convertToArrayFirst, convert_to_array_first)
        if weights is None:
            weights = [1.0 for _ in input_cols]

        assert len(input_cols) == len(weights)

        self.set(self.weights, weights)
        self.set(self.dimNum, dim_num)

    def get_task_name(self) -> str:
        return self.getOrDefault(self.taskName)

    def get_remove_cols(self) -> List[str]:
        return self.getOrDefault(self.removeCols)

    def get_convert_to_array_first(self) -> bool:
        return self.getOrDefault(self.convertToArrayFirst)

    def get_weights(self) -> List[int]:
        return self.getOrDefault(self.weights)

    def get_dim_num(self) -> int:
        return self.getOrDefault(self.dimNum)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        logger.debug(f"In {type(self)}. Columns: {sorted(dataset.columns)}")

        pred_cols = self.getInputCols()

        if self.get_task_name() in ["binary", "multiclass"]:
            out_col = vector_averaging(sf.array(*pred_cols), sf.lit(self.get_dim_num()))
        else:
            out_col = scalar_averaging(sf.array(*pred_cols))

        cols_to_remove = set(self.get_remove_cols())
        cols_to_select = [c for c in dataset.columns if c not in cols_to_remove]
        out_df = dataset.select(*cols_to_select, out_col.alias(self.getOutputCol()))

        logger.debug(f"Out {type(self)}. Columns: {sorted(out_df.columns)}")

        return out_df
