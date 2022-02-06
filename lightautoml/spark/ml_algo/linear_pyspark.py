"""Linear models for tabular datasets."""

import logging
from copy import copy
from typing import Tuple, Optional, List
from typing import Union

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

from pyspark.sql import functions as F

from .base import TabularMLAlgo, SparkMLModel
from ..dataset.base import SparkDataset, SparkDataFrame
from ...utils.timer import TaskTimer
from ...utils.tmp_utils import log_data


import numpy as np

logger = logging.getLogger(__name__)

LinearEstimator = Union[LogisticRegression, LinearRegression]
LinearEstimatorModel = Union[LogisticRegressionModel, LinearRegressionModel]


class LinearLBFGS(TabularMLAlgo):

    _name: str = "LinearL2"

    _default_params = {
        "tol": 1e-6,
        "maxIter": 100,
        "regParam":
        [
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

    def __init__(self,
                 default_params: Optional[dict] = None,
                 freeze_defaults: bool = True,
                 timer: Optional[TaskTimer] = None,
                 optimization_search_space: Optional[dict] = {}):
        super().__init__()

        self._prediction_col = f"prediction_{self._name}"
        self.task = None
        self._timer = timer

    def _infer_params(self, train: SparkDataset) -> Tuple[List[Tuple[float, Pipeline]], int]:
        logger.debug("Building pipeline in linear lGBFS")
        params = copy(self.params)

        # categorical features
        cat_feats = [feat for feat in train.features if train.roles[feat].name == "Category"]
        non_cat_feats = [feat for feat in train.features if train.roles[feat].name != "Category"]

        ohe = OneHotEncoder(inputCols=cat_feats, outputCols=[f"{f}_{self._name}_ohe" for f in cat_feats])
        assembler = VectorAssembler(
            inputCols=non_cat_feats + ohe.getOutputCols(),
            outputCol=f"{self._name}_vassembler_features"
        )

        if "regParam" in params:
            reg_params = params["regParam"]
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
            # TODO: SPARK-LAMA add params processing later
            if self.task.name in ["binary", "multiclass"]:
                model = LogisticRegression(featuresCol=assembler.getOutputCol(),
                                           labelCol=train.target_column,
                                           predictionCol=self._prediction_col,
                                           **instance_params)
            elif self.task.name == "reg":
                model = LinearRegression(featuresCol=assembler.getOutputCol(),
                                         labelCol=train.target_column,
                                         predictionCol=self._prediction_col,
                                         **instance_params)
                model.setSolver("l-bfgs")
            else:
                raise ValueError("Task not supported")

            pipeline = Pipeline(stages=[ohe, assembler, model])

            return pipeline

        estimators = [(rp, build_pipeline(rp)) for rp in reg_params]

        return estimators, es

    def fit_predict_single_fold(
        self, train: SparkDataset, valid: SparkDataset
    ) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        logger.info(f"fit_predict single fold in LinearLBGFS. Num of features: {len(train.features)} ")
        log_data("spark_linear_l2_train_val", {"train": train.to_pandas(), "valid": valid.to_pandas()})

        if self.task is None:
            self.task = train.task

        # TODO: SPARK-LAMA target column?
        train_sdf = self._make_sdf_with_target(train)
        val_sdf = self._make_sdf_with_target(valid)

        estimators, early_stopping = self._infer_params(train)

        assert len(estimators) > 0

        es: int = 0
        best_score: float = -np.inf

        best_model: Optional[SparkMLModel] = None
        best_val_pred: Optional[SparkDataFrame] = None
        for rp, pipeline in estimators:
            logger.debug(f"Fitting estimators with regParam {rp}")
            ml_model = pipeline.fit(train_sdf)
            val_pred = ml_model.transform(val_sdf)
            preds_to_score = val_pred.select(
                F.col(self._prediction_col).alias("prediction"),
                F.col(valid.target_column).alias("target")
            )
            current_score = self.score(preds_to_score)
            if current_score > best_score:
                best_score = current_score
                best_model = ml_model
                best_val_pred = val_pred
                es = 0
            else:
                es += 1

            if es >= early_stopping:
                break

        return best_model, best_val_pred, self._prediction_col

    def predict_single_fold(self, dataset: SparkDataset, model: SparkMLModel) -> SparkDataFrame:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: ``SparkDataset`` used for prediction.

        Returns:
            Predictions for input dataset.

        """
        log_data("spark_linear_l2_predict", {"predict": dataset.to_pandas()})
        pred = model.transform(dataset.data)
        return pred
