import logging
import multiprocessing
from copy import copy
from typing import Callable, Dict, Optional, Tuple, Union, cast

import pandas as pd
from pandas import Series
from pyspark.ml import Transformer, PipelineModel
from pyspark.ml.feature import VectorAssembler
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor

from lightautoml.ml_algo.tuning.base import Distribution, SearchSpace
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.ml_algo.base import SparkTabularMLAlgo, SparkMLModel, AveragingTransformer
from lightautoml.spark.transformers.base import DropColumnsTransformer
from lightautoml.spark.validation.base import SparkBaseTrainValidIterator
from lightautoml.utils.timer import TaskTimer
from lightautoml.validation.base import TrainValidIterator

logger = logging.getLogger(__name__)


class SparkBoostLGBM(SparkTabularMLAlgo, ImportanceEstimator):

    _name: str = "LightGBM"

    _default_params = {
        # "improvementTolerance": 1e-4,
        "learningRate": 0.05,
        "numLeaves": 128,
        "featureFraction": 0.7,
        "baggingFraction": 0.7,
        "baggingFreq": 1,
        "maxDepth": -1,
        "minGainToSplit": 0.0,
        "maxBin": 255,
        "minDataInLeaf": 5,
        # e.g. num trees
        "numIterations": 3000,
        "earlyStoppingRound": 50,
        # for regression
        "alpha": 1.0,
        "lambdaL1": 0.0,
        "lambdaL2": 0.0,
        # seeds
        # "baggingSeed": 42
    }

    # mapping between metric name defined via SparkTask
    # and metric names supported by LightGBM
    _metric2lgbm = {
        "binary": {
            "auc": "auc",
            "aupr": "areaUnderPR"
        },
        "reg": {
            "r2": "rmse",
            "mse": "mse",
            "mae": "mae",
        },
        "multiclass": {
            "crossentropy": "cross_entropy"
        }
    }

    def __init__(self,
                 cacher_key: str,
                 default_params: Optional[dict] = None,
                 freeze_defaults: bool = True,
                 timer: Optional[TaskTimer] = None,
                 optimization_search_space: Optional[dict] = {}):
        SparkTabularMLAlgo.__init__(self, cacher_key, default_params, freeze_defaults, timer, optimization_search_space)
        self._probability_col_name = "probability"
        self._prediction_col_name = "prediction"
        self._raw_prediction_col_name = "raw_prediction"
        self._assembler = None
        self._drop_cols_transformer = None

    def _infer_params(self) -> Tuple[dict, int]:
        """Infer all parameters in lightgbm format.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval).
            About parameters: https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/engine.html

        """
        assert self.task is not None

        task = self.task.name

        params = copy(self.params)

        if "isUnbalance" in params:
            params["isUnbalance"] = True if params["isUnbalance"] == 1 else False

        verbose_eval = 1

        if task == "reg":
            params["objective"] = "regression"
            params["metric"] = self._metric2lgbm[task].get(self.task.metric_name, None)
        elif task == "binary":
            params["objective"] = "binary"
            params["metric"] = self._metric2lgbm[task].get(self.task.metric_name, None)
        elif task == "multiclass":
            params["objective"] = "multiclass"
            params["metric"] = "multiclass"
        else:
            raise ValueError(f"Unsupported task type: {task}")

        if task != "reg":
            if "alpha" in params:
                del params["alpha"]
            if "lambdaL1" in params:
                del params["lambdaL1"]
            if "lambdaL2" in params:
                del params["lambdaL2"]

        params = {**params}

        return params, verbose_eval

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:
        self.task = train_valid_iterator.train.task

        sds = cast(SparkDataset, train_valid_iterator.train)
        rows_num = sds.data.count()
        task = train_valid_iterator.train.task.name

        suggested_params = copy(self.default_params)

        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        if task == "reg":
            suggested_params = {
                "learningRate": 0.05,
                "numLeaves": 32,
                "featureFraction": 0.9,
                "baggingFraction": 0.9,
            }

        if rows_num <= 10000:
            init_lr = 0.01
            ntrees = 3000
            es = 200

        elif rows_num <= 20000:
            init_lr = 0.02
            ntrees = 3000
            es = 200

        elif rows_num <= 100000:
            init_lr = 0.03
            ntrees = 2000
            es = 200
        elif rows_num <= 300000:
            init_lr = 0.04
            ntrees = 2000
            es = 100
        else:
            init_lr = 0.05
            ntrees = 2000
            es = 100

        if rows_num > 300000:
            suggested_params["numLeaves"] = 128 if task == "reg" else 244
        elif rows_num > 100000:
            suggested_params["numLeaves"] = 64 if task == "reg" else 128
        elif rows_num > 50000:
            suggested_params["numLeaves"] = 32 if task == "reg" else 64
            # params['reg_alpha'] = 1 if task == 'reg' else 0.5
        elif rows_num > 20000:
            suggested_params["numLeaves"] = 32 if task == "reg" else 32
            suggested_params["alpha"] = 0.5 if task == "reg" else 0.0
        elif rows_num > 10000:
            suggested_params["numLeaves"] = 32 if task == "reg" else 64
            suggested_params["alpha"] = 0.5 if task == "reg" else 0.2
        elif rows_num > 5000:
            suggested_params["numLeaves"] = 24 if task == "reg" else 32
            suggested_params["alpha"] = 0.5 if task == "reg" else 0.5
        else:
            suggested_params["numLeaves"] = 16 if task == "reg" else 16
            suggested_params["alpha"] = 1 if task == "reg" else 1

        suggested_params["learningRate"] = init_lr
        suggested_params["numIterations"] = ntrees
        suggested_params["earlyStoppingRound"] = es

        if task != "reg":
            if "alpha" in suggested_params:
                del suggested_params["alpha"]

        return suggested_params

    def _get_default_search_spaces(self, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        """Train on train dataset and predict on holdout dataset.

        Args:
            fold_prediction_column: column name for predictions made for this fold
            full: Full dataset that include train and valid parts and a bool column that delimits records
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        assert self.task is not None

        optimization_search_space = dict()

        optimization_search_space["featureFraction"] = SearchSpace(
            Distribution.UNIFORM,
            low=0.5,
            high=1.0,
        )

        optimization_search_space["numLeaves"] = SearchSpace(
            Distribution.INTUNIFORM,
            low=4,
            high=255,
        )

        if self.task.name == "binary" or self.task.name == "multiclass":
            optimization_search_space["isUnbalance"] = SearchSpace(
                Distribution.DISCRETEUNIFORM,
                low=0,
                high=1,
                q=1
            )

        if estimated_n_trials > 30:
            optimization_search_space["baggingFraction"] = SearchSpace(
                Distribution.UNIFORM,
                low=0.5,
                high=1.0,
            )

            optimization_search_space["minSumHessianInLeaf"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-3,
                high=10.0,
            )

        if estimated_n_trials > 100:
            if self.task.name == "reg":
                optimization_search_space["alpha"] = SearchSpace(
                    Distribution.LOGUNIFORM,
                    low=1e-8,
                    high=10.0,
                )

            optimization_search_space["lambdaL1"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-8,
                high=10.0,
            )

        return optimization_search_space

    def predict_single_fold(self,
                            dataset: SparkDataset,
                            model: Union[LightGBMRegressor, LightGBMClassifier]) -> SparkDataFrame:

        temp_sdf = self._assembler.transform(dataset.data)

        pred = model.transform(temp_sdf)

        return pred

    def fit_predict_single_fold(self,
                                fold_prediction_column: str,
                                full: SparkDataset,
                                train: SparkDataset,
                                valid: SparkDataset) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        assert self.validation_column in full.data.columns, 'Train should contain validation column'

        if self.task is None:
            self.task = full.task

        (
            params,
            verbose_eval,
            fobj,
            feval,
        ) = self._infer_params()

        logger.info(f"Input cols for the vector assembler: {full.features}")
        logger.info(f"Running lgb with the following params: {params}")

        # TODO: SPARK-LAMA reconsider using of 'keep' as a handleInvalid value
        if self._assembler is None:
            self._assembler = VectorAssembler(
                inputCols=self.input_features,
                outputCol=f"{self._name}_vassembler_features",
                handleInvalid="keep"
            )

        LGBMBooster = LightGBMRegressor if full.task.name == "reg" else LightGBMClassifier

        if full.task.name == 'binary':
            params['rawPredictionCol'] = fold_prediction_column
            params['probabilityCol'] = self._probability_col_name
            params['predictionCol'] = self._prediction_col_name
        elif full.task.name == 'multiclass':
            params['rawPredictionCol'] = self._raw_prediction_col_name
            params['probabilityCol'] = fold_prediction_column
            params['predictionCol'] = self._prediction_col_name
        else:
            params['predictionCol'] = fold_prediction_column

        master_addr = train.spark_session.conf.get('spark.master')
        if master_addr.startswith('local'):
            cores_str = master_addr[len("local["):-1]
            cores = int(cores_str) if cores_str != "*" else multiprocessing.cpu_count()
            params["numThreads"] = max(cores - 1, 1)
        else:
            params["numThreads"] = max(int(train.spark_session.conf.get("spark.executor.cores", "1")) - 1, 1)

        lgbm = LGBMBooster(
            **params,
            featuresCol=self._assembler.getOutputCol(),
            labelCol=full.target_column,
            validationIndicatorCol=self.validation_column,
            verbosity=verbose_eval,
            useSingleDatasetMode=True,
            isProvideTrainingMetric=True
        )

        logger.info(f"Use single dataset mode: {lgbm.getUseSingleDatasetMode()}. NumThreads: {lgbm.getNumThreads()}")

        if full.task.name == "reg":
            lgbm.setAlpha(0.5).setLambdaL1(0.0).setLambdaL2(0.0)

        temp_sdf = self._assembler.transform(full.data)

        ml_model = lgbm.fit(temp_sdf)

        val_pred = ml_model.transform(self._assembler.transform(valid.data))
        val_pred = DropColumnsTransformer(
            remove_cols=[],
            optional_remove_cols=[self._prediction_col_name, self._probability_col_name]
        ).transform(val_pred)

        return ml_model, val_pred, fold_prediction_column

    def fit(self, train_valid: SparkBaseTrainValidIterator):
        self.fit_predict(train_valid)

    def get_features_score(self) -> Series:
        imp = 0
        for model in self.models:
            imp = imp + pd.Series(model.getFeatureImportances(importance_type='gain'))

        imp = imp / len(self.models)

        result = Series(list(imp), index=self.features).sort_values(ascending=False)
        return result

    def _build_transformer(self) -> Transformer:
        avr = self._build_averaging_transformer()
        models = [el for m in self.models for el in [m, DropColumnsTransformer(
            remove_cols=[],
            optional_remove_cols=[self._prediction_col_name, self._probability_col_name, self._raw_prediction_col_name]
        )]]
        averaging_model = PipelineModel(stages=[self._assembler] + models + [avr])
        return averaging_model

    def _build_averaging_transformer(self) -> Transformer:
        avr = AveragingTransformer(
            self.task.name,
            input_cols=self._models_prediction_columns,
            output_col=self.prediction_feature,
            remove_cols=[self._assembler.getOutputCol()] + self._models_prediction_columns,
            convert_to_array_first=not (self.task.name == "reg"),
            dim_num=self.n_classes
        )
        return avr

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
        self.timer.start()

        self.input_roles = train_valid_iterator.input_roles
        
        return super().fit_predict(train_valid_iterator)
