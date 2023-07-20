"""Linear models for tabular datasets."""

import logging

from copy import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from lightautoml.ml_algo.tuning.base import Distribution
from lightautoml.ml_algo.tuning.base import SearchSpace
from lightautoml.utils.timer import TaskTimer
from pyspark.ml import Estimator
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml import Transformer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
from pyspark.sql import functions as sf

from sparklightautoml.ml_algo.base import AveragingTransformer
from sparklightautoml.ml_algo.base import ComputationalParameters
from sparklightautoml.ml_algo.base import SparkMLModel
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo
from sparklightautoml.validation.base import SparkBaseTrainValidIterator
from sparklightautoml.validation.base import split_out_train
from sparklightautoml.validation.base import split_out_val

from ..dataset.base import SparkDataset
from ..transformers.base import DropColumnsTransformer
from ..utils import SparkDataFrame
from ..utils import log_exception


logger = logging.getLogger(__name__)

LinearEstimator = Union[LogisticRegression, LinearRegression]
LinearEstimatorModel = Union[LogisticRegressionModel, LinearRegressionModel]


class SparkLinearLBFGS(SparkTabularMLAlgo):
    """LBFGS L2 regression based on Spark MLlib.


    default_params:

        - tol: The tolerance for the stopping criteria.
        - maxIter: Maximum iterations of L-BFGS.
        - aggregationDepth: Param for suggested depth for treeAggregate.
        - elasticNetParam: Elastic net parameter.
        - regParam: Regularization parameter.
        - early_stopping: Maximum rounds without improving.

    freeze_defaults:

        - ``True`` :  params may be rewrited depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

    _name: str = "LinearL2"

    _default_params = {
        "tol": 1e-6,
        "maxIter": 100,
        "aggregationDepth": 2,
        "elasticNetParam": 0.7,
        "regParam": [
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2,
            5e-2,
            1e-1,
            5e-1,
            1,
            5,
            10,
            50,
            100,
            500,
            1000,
            5000,
            10000,
            50000,
            100000,
        ],
        "early_stopping": 2,
    }

    def __init__(
        self,
        default_params: Optional[dict] = None,
        freeze_defaults: bool = True,
        timer: Optional[TaskTimer] = None,
        optimization_search_space: Optional[dict] = None,
        persist_output_dataset: bool = True,
        computations_settings: Optional[ComputationalParameters] = None,
    ):
        optimization_search_space = optimization_search_space if optimization_search_space else dict()
        super().__init__(
            default_params,
            freeze_defaults,
            timer,
            optimization_search_space,
            persist_output_dataset,
            computations_settings,
        )

        self._prediction_col = f"prediction_{self._name}"
        self.task = None
        self._timer = timer
        # self._ohe = None
        self._assembler = None

        self._raw_prediction_col_name = "raw_prediction"
        self._probability_col_name = "probability"
        self._prediction_col_name = "prediction"

    def _get_default_search_spaces(self, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        """Train on train dataset and predict on holdout dataset.

        Args:.
            suggested_params: suggested params
            estimated_n_trials: Number of trials.

        Returns:
            Target predictions for valid dataset.

        """
        optimization_search_space = dict()
        optimization_search_space["regParam"] = SearchSpace(
            Distribution.UNIFORM,
            low=1e-5,
            high=100000,
        )

        return optimization_search_space

    def _infer_params(
        self, train: SparkDataset, fold_prediction_column: str
    ) -> Tuple[List[Tuple[float, Estimator]], int]:
        logger.debug("Building pipeline in linear lGBFS")
        params = copy(self.params)

        if "regParam" in params:
            reg_params = params["regParam"] if isinstance(params["regParam"], list) else [params["regParam"]]
            del params["regParam"]
        else:
            reg_params = [1.0]

        if "early_stopping" in params:
            es = params["early_stopping"]
            del params["early_stopping"]
        else:
            es = 100

        def build_pipeline(reg_param: int):
            instance_params = copy(params)
            instance_params["regParam"] = reg_param
            if self.task.name in ["binary", "multiclass"]:
                model = LogisticRegression(
                    featuresCol=self._assembler.getOutputCol(),
                    labelCol=train.target_column,
                    probabilityCol=fold_prediction_column,
                    rawPredictionCol=self._raw_prediction_col_name,
                    predictionCol=self._prediction_col_name,
                    **instance_params,
                )
            elif self.task.name == "reg":
                model = LinearRegression(
                    featuresCol=self._assembler.getOutputCol(),
                    labelCol=train.target_column,
                    predictionCol=fold_prediction_column,
                    **instance_params,
                )
                model = model.setSolver("l-bfgs")
            else:
                raise ValueError("Task not supported")

            return model

        estimators = [(rp, build_pipeline(rp)) for rp in reg_params]

        return estimators, es

    def fit_predict_single_fold(
        self,
        fold_prediction_column: str,
        validation_column: str,
        train: SparkDataset,
        runtime_settings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        logger.info(f"fit_predict single fold in LinearLBGFS. Num of features: {len(self.input_roles.keys())} ")

        if self.task is None:
            self.task = train.task

        train_sdf = split_out_train(train.data, validation_column)
        val_sdf = split_out_val(train.data, validation_column)

        estimators, early_stopping = self._infer_params(train, fold_prediction_column)

        assert len(estimators) > 0

        es: int = 0
        best_score: float = -np.inf

        best_model: Optional[SparkMLModel] = None
        best_val_pred: Optional[SparkDataFrame] = None
        for rp, model in estimators:
            logger.debug(f"Fitting estimators with regParam {rp}")
            pipeline = Pipeline(stages=[self._assembler, model])
            ml_model = pipeline.fit(train_sdf)
            val_pred = ml_model.transform(val_sdf)
            preds_to_score = val_pred.select(
                sf.col(fold_prediction_column).alias("prediction"), sf.col(train.target_column).alias("target")
            )
            current_score = self.score(preds_to_score)
            if current_score > best_score:
                best_score = current_score
                best_model = ml_model.stages[-1]
                best_val_pred = val_pred
                es = 0
            else:
                es += 1

            if es >= early_stopping:
                break

        logger.info("fit_predict single fold finished in LinearLBGFS")

        return best_model, best_val_pred, fold_prediction_column

    def predict_single_fold(self, dataset: SparkDataset, model: SparkMLModel) -> SparkDataFrame:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: ``SparkDataset`` used for prediction.

        Returns:
            Predictions for input dataset.

        """
        pred = model.transform(dataset.data)
        return pred

    def _build_transformer(self) -> Transformer:
        avr = self._build_averaging_transformer()
        models = [
            el
            for m in self.models
            for el in [
                m,
                DropColumnsTransformer(
                    remove_cols=[],
                    optional_remove_cols=[
                        self._prediction_col_name,
                        self._probability_col_name,
                        self._raw_prediction_col_name,
                    ],
                ),
            ]
        ]
        averaging_model = PipelineModel(
            stages=[
                self._assembler,
                *models,
                avr,
                self._build_vector_size_hint(self.prediction_feature, self.prediction_role),
            ]
        )
        return averaging_model

    def _build_averaging_transformer(self) -> Transformer:
        avr = AveragingTransformer(
            self.task.name,
            input_cols=self._models_prediction_columns,
            output_col=self.prediction_feature,
            remove_cols=[self._assembler.getOutputCol(), *self._models_prediction_columns],
            convert_to_array_first=not (self.task.name == "reg"),
            dim_num=self.n_classes,
        )
        return avr

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
        logger.info("Starting LinearLGBFS")
        self.timer.start()

        cat_feats = [feat for feat, role in train_valid_iterator.train.roles.items() if role.name == "Category"]
        non_cat_feats = [feat for feat, role in train_valid_iterator.train.roles.items() if role.name != "Category"]

        self._assembler = VectorAssembler(
            inputCols=non_cat_feats + cat_feats,
            outputCol=f"{self._name}_vassembler_features",
        )

        result = super().fit_predict(train_valid_iterator)

        logger.info("LinearLGBFS is finished")

        return result
