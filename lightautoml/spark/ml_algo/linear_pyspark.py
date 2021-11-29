"""Linear models for tabular datasets."""

import logging
from copy import copy
from typing import Tuple
from typing import Union

from pyspark.ml import Pipeline, Model
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

from .base import TabularMLAlgo
from ..dataset.base import SparkDataset, SparkDataFrame

logger = logging.getLogger(__name__)

LinearEstimator = Union[LogisticRegression, LinearRegression]
LinearEstimatorModel = Union[LogisticRegressionModel, LinearRegressionModel]


class LinearLBFGS(TabularMLAlgo):

    _name: str = "LinearL2"

    def __init__(self,
                 params={}):
        self._prediction_col = f"prediction_{self._name}"
        self.params = params
        self.task = None

        super().__init__()

    def _infer_params(self, train: SparkDataset) -> Pipeline:
        params = copy(self.params)

        # categorical features
        cat_feats = [feat for feat in train.features if train.roles[feat].name == "Category"]
        non_cat_feats = [feat for feat in train.features if train.roles[feat].name != "Category"]

        ohe = OneHotEncoder(inputCols=cat_feats, outputCols=[f"{f}_{self._name}_ohe" for f in cat_feats])
        assembler = VectorAssembler(
            inputCols=non_cat_feats + ohe.getOutputCols(),
            outputCol=f"{self._name}_vassembler_features"
        )

        if self.task.name in ["binary", "multiclass"]:
            model = LogisticRegression(featuresCol=assembler.getOutputCol(),
                                       labelCol=train.target_column,
                                       predictionCol=self._prediction_col,
                                       **params)
        elif self.task.name == "reg":
            model = LinearRegression(featuresCol=assembler.getOutputCol(),
                                     labelCol=train.target_column,
                                     predictionCol=self._prediction_col,
                                     **params)
            model.setSolver("l-bfgs")
        else:
            raise ValueError("Task not supported")

        pipeline = Pipeline(stages=[ohe, assembler, model])

        return pipeline

    def fit_predict_single_fold(
        self, train: SparkDataset, valid: SparkDataset
    ) -> Tuple[Model, SparkDataFrame, str]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        if self.task is None:
            self.task = train.task

        # TODO: SPARK-LAMA target column?
        train_sdf = self._make_sdf_with_target(train.data)
        val_sdf = valid.data

        pipeline = self._infer_params(train)
        ml_model = pipeline.fit(train_sdf)

        val_pred = ml_model.transform(val_sdf)

        return ml_model, val_pred, self._prediction_col

    def predict_single_fold(self, dataset: SparkDataset, model: Model) -> SparkDataFrame:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: ``SparkDataset`` used for prediction.

        Returns:
            Predictions for input dataset.

        """
        pred = model.transform(dataset.data)
        return pred
