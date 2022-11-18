import functools
import logging
from copy import copy
from typing import Tuple, cast, List, Optional, Sequence, Union

import numpy as np
from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import NumericRole
from lightautoml.ml_algo.base import MLAlgo
from lightautoml.utils.timer import TaskTimer
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml.param import Params
from pyspark.ml.param.shared import HasInputCols, HasOutputCol, Param
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.sql import functions as sf
from pyspark.sql.types import IntegerType

from sparklightautoml.dataset.base import SparkDataset, PersistenceLevel
from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.pipelines.base import TransformerInputOutputRoles
from sparklightautoml.utils import SparkDataFrame, log_exception
from sparklightautoml.validation.base import SparkBaseTrainValidIterator

logger = logging.getLogger(__name__)

SparkMLModel = PipelineModel


class SparkTabularMLAlgo(MLAlgo, TransformerInputOutputRoles):
    """Machine learning algorithms that accepts numpy arrays as input."""

    _name: str = "SparkTabularMLAlgo"
    _default_validation_col_name: str = SparkBaseTrainValidIterator.TRAIN_VAL_COLUMN

    def __init__(
        self,
        default_params: Optional[dict] = None,
        freeze_defaults: bool = True,
        timer: Optional[TaskTimer] = None,
        optimization_search_space: Optional[dict] = None,
    ):
        optimization_search_space = optimization_search_space if optimization_search_space else dict()
        super().__init__(default_params, freeze_defaults, timer, optimization_search_space)
        self.n_classes: Optional[int] = None
        # names of columns that should contain predictions of individual models
        self._models_prediction_columns: Optional[List[str]] = None

        self._prediction_role: Optional[Union[NumericRole, NumericVectorOrArrayRole]] = None
        self._input_roles: Optional[RolesDict] = None
        self._service_columns: Optional[List[str]] = None

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

        preds_dfs: List[SparkDataFrame] = []
        self._models_prediction_columns = []

        with train_valid_iterator.frozen() as frozen_train_valid_iterator:
            for n, (train, valid) in enumerate(frozen_train_valid_iterator):
                if iterator_len > 1:
                    logger.info2(
                        "===== Start working with \x1b[1mfold {}\x1b[0m for \x1b[1m{}\x1b[0m "
                        "=====".format(n, self._name)
                    )
                self.timer.set_control_point()

                model_prediction_col = f"{self.prediction_feature}_{n}"
                model, val_pred, _ = self.fit_predict_single_fold(model_prediction_col, train, valid)
                val_pred = val_pred.select(SparkDataset.ID_COLUMN, train.target_column, model_prediction_col)

                self._models_prediction_columns.append(model_prediction_col)
                self.models.append(model)
                preds_dfs.append(val_pred)

                self.timer.write_run_info()

                if (n + 1) != len(frozen_train_valid_iterator):
                    # split into separate cases because timeout checking affects parent pipeline timer
                    if self.timer.time_limit_exceeded():
                        logger.info("Time limit exceeded after calculating fold {0}\n".format(n))
                        break

        full_preds_df = self._combine_val_preds(train_valid_iterator.get_validation_data(), preds_dfs)
        full_preds_df = self._build_averaging_transformer().transform(full_preds_df)
        full_preds_df = self._build_vector_size_hint(self.prediction_feature, self._prediction_role)\
            .transform(full_preds_df)

        pred_ds = valid_ds.empty()
        pred_ds.set_data(
            full_preds_df,
            list(self.output_roles.keys()),
            self.output_roles,
            dependencies=[train_valid_iterator],
            name=f"{self._name}"
        )
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
        self, fold_prediction_column: str, train: SparkDataset, valid: SparkDataset
    ) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            fold_prediction_column: column name for predictions made for this fold
            train: Train Dataset.
            valid: Validation Dataset.

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
                outp_dim, f"{self.prediction_feature}" + "_{}", np.float32, force_input=True, prob=True
            )
        elif self.task.name == "binary":
            outp_dim = 2
            self._prediction_role = NumericVectorOrArrayRole(
                outp_dim, f"{self.prediction_feature}" + "_{}", np.float32, force_input=True, prob=True
            )
        else:
            self._prediction_role = NumericRole(np.float32, force_input=True, prob=False)

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
        output.set_data(preds, preds.columns, roles, name="single_prediction_dataset")

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
            lambda acc, x: acc.join(x, on=SparkDataset.ID_COLUMN, how='left'),
            [val_pred.drop(val_data.target_column) for val_pred in val_preds],
            initial_df
        )

        return full_val_preds


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
        weights = {c: w for w, c in zip(self.get_weights(), pred_cols)}
        non_null_count_col = sf.lit(len(pred_cols)) - sum(sf.isnull(c).astype(IntegerType()) for c in pred_cols)

        if self.get_task_name() in ["binary", "multiclass"]:

            def convert_column(c):
                return vector_to_array(c).alias(c) if self.get_convert_to_array_first() else sf.col(c)

            normalized_cols = [
                sf.when(sf.isnull(c), sf.array(*[sf.lit(0.0) for _ in range(self.get_dim_num())]))
                .otherwise(convert_column(c))
                .alias(c)
                for c in pred_cols
            ]
            arr_fields_summ = sf.transform(
                sf.arrays_zip(*normalized_cols),
                lambda x: sf.aggregate(
                    sf.array(*[x[c] * sf.lit(weights[c]) for c in pred_cols]), sf.lit(0.0), lambda acc, y: acc + y
                )
                / non_null_count_col,
            )

            out_col = array_to_vector(arr_fields_summ) if self.get_convert_to_array_first() else arr_fields_summ
        else:
            scalar_fields_summ = (
                sf.aggregate(
                    sf.array(*[sf.col(c) * sf.lit(weights[c]) for c in pred_cols]),
                    sf.lit(0.0),
                    lambda acc, x: acc + sf.when(sf.isnull(x), sf.lit(0.0)).otherwise(x),
                )
                / non_null_count_col
            )

            out_col = scalar_fields_summ

        cols_to_remove = set(self.get_remove_cols())
        cols_to_select = [c for c in dataset.columns if c not in cols_to_remove]
        out_df = dataset.select(*cols_to_select, out_col.alias(self.getOutputCol()))

        logger.debug(f"Out {type(self)}. Columns: {sorted(out_df.columns)}")

        return out_df
