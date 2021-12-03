"""Wrapped LightGBM for tabular datasets."""

import logging

from contextlib import redirect_stdout
from copy import copy
from typing import Callable, Dict, Optional, Tuple, List, Union

import lightgbm as lgb
import numpy as np

from pandas import Series

from dataset.roles import CategoryRole
from lightautoml.utils.logging import LoggerStream
from lightautoml.ml_algo.tuning.base import Distribution, SearchSpace

# TODO SPARK-LAMA: How to understand which of these we have to use?
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor

from lightautoml.spark.pipelines.selection.base import SparkImportanceEstimator
from lightautoml.spark.validation.base import TrainValidIterator
from lightautoml.spark.ml_algo.base import SparkTabularDataset, SparkTabularMLAlgo

from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F


logger = logging.getLogger(__name__)


class BoostLGBM(SparkTabularMLAlgo, SparkImportanceEstimator):

    models: List[Union[LightGBMClassifier, LightGBMRegressor]]

    _name: str = "LightGBM"

    _default_params = {
        "task": "train",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "max_depth": -1,
        "verbosity": -1,
        "reg_alpha": 1,
        "reg_lambda": 0.0,
        "min_split_gain": 0.0,
        "zero_as_missing": False,
        "num_threads": 4,
        "max_bin": 255,
        "min_data_in_bin": 3,
        "num_trees": 3000,
        "early_stopping_rounds": 100,
        "random_state": 42,
    }

    def _infer_params(self) -> Tuple[dict, int, int, int, Optional[Callable], Optional[Callable]]:
        """Infer all parameters in lightgbm format.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval).
            About parameters: https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/engine.html

        """
        # TODO: Check how it works with custom tasks
        params = copy(self.params)
        early_stopping_rounds = params.pop("early_stopping_rounds")
        num_trees = params.pop("num_trees")

        verbose_eval = True

        # get objective params
        # TODO SPARK-LAMA: Only for smoke test
        loss = None  # self.task.losses["lgb"]
        params["objective"] = None  # loss.fobj_name
        fobj = None  # loss.fobj

        # get metric params
        params["metric"] = None  # loss.metric_name
        feval = None  # loss.feval

        params["num_class"] = None  # self.n_classes
        # add loss and tasks params if defined
        # params = {**params, **loss.fobj_params, **loss.metric_params}
        params = {**params}

        return params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:

        # TODO SPARK-LAMA: Only for smoke test
        try:
            is_reg = train_valid_iterator.train.task.name == "reg"
        except AttributeError:
            is_reg = False

        suggested_params = copy(self.default_params)

        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        if is_reg:
            suggested_params = {
                "learning_rate": 0.05,
                "num_leaves": 32,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
            }

        suggested_params["num_leaves"] = 128 if is_reg else 244

        suggested_params["learning_rate"] = 0.05
        suggested_params["num_trees"] = 2000
        suggested_params["early_stopping_rounds"] = 100

        return suggested_params

    def _get_default_search_spaces(self, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        """Sample hyperparameters from suggested.

        Args:
            trial: Optuna trial object.
            suggested_params: Dict with parameters.
            estimated_n_trials: Maximum number of hyperparameter estimations.

        Returns:
            dict with sampled hyperparameters.

        """
        optimization_search_space = {}

        optimization_search_space["feature_fraction"] = SearchSpace(
            Distribution.UNIFORM,
            low=0.5,
            high=1.0,
        )

        optimization_search_space["num_leaves"] = SearchSpace(
            Distribution.INTUNIFORM,
            low=16,
            high=255,
        )

        if estimated_n_trials > 30:
            optimization_search_space["bagging_fraction"] = SearchSpace(
                Distribution.UNIFORM,
                low=0.5,
                high=1.0,
            )

            optimization_search_space["min_sum_hessian_in_leaf"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-3,
                high=10.0,
            )

        if estimated_n_trials > 100:
            optimization_search_space["reg_alpha"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-8,
                high=10.0,
            )
            optimization_search_space["reg_lambda"] = SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-8,
                high=10.0,
            )

        return optimization_search_space

    # def fit_predict_single_fold(self,
    #                             train: SparkTabularDataset,
    #                             valid: SparkTabularDataset) -> Tuple[lgb.Booster, np.ndarray]:
    #
    #     (
    #         params,
    #         num_trees,
    #         early_stopping_rounds,
    #         verbose_eval,
    #         fobj,
    #         feval,
    #     ) = self._infer_params()
    #
    #     train_target, train_weight = self.task.losses["lgb"].fw_func(train.target, train.weights)
    #     valid_target, valid_weight = self.task.losses["lgb"].fw_func(valid.target, valid.weights)
    #
    #     lgb_train = lgb.Dataset(train.data, label=train_target, weight=train_weight)
    #     lgb_valid = lgb.Dataset(valid.data, label=valid_target, weight=valid_weight)
    #
    #     with redirect_stdout(LoggerStream(logger, verbose_eval=100)):
    #         model = lgb.train(
    #             params,
    #             lgb_train,
    #             num_boost_round=num_trees,
    #             valid_sets=[lgb_valid],
    #             valid_names=["valid"],
    #             fobj=fobj,
    #             feval=feval,
    #             early_stopping_rounds=early_stopping_rounds,
    #             verbose_eval=verbose_eval,
    #         )
    #
    #     val_pred = model.predict(valid.data)
    #     val_pred = self.task.losses["lgb"].bw_func(val_pred)
    #
    #     return model, val_pred

    # def predict_single_fold(self,
    #                         model: Union[LightGBMRegressor, LightGBMClassifier],
    #                         dataset: SparkTabularDataset) -> np.ndarray:
    #
    #     # TODO SPARK-LAMA: Set target columns and so on.
    #     pred = self.task.losses["lgb"].bw_func(model.predict(dataset.data))
    #
    #     return pred

    # def get_features_score(self) -> Series:
    #     """Computes feature importance as mean values of feature importance provided by lightgbm per all models.
    #
    #     Returns:
    #         Series with feature importances.
    #
    #     """
    #
    #     imp = 0
    #     for model in self.models:
    #         imp = imp + model.getFeatureImportance(importance_type="gain")
    #
    #     imp = imp / len(self.models)
    #
    #     return Series(imp, index=self.features).sort_values(ascending=False)

    def fit_predict(self, train_valid_iterator: TrainValidIterator) -> SparkTabularDataset:

        # TODO SPARK-LAMA: Only for smoke test
        try:
            is_reg = train_valid_iterator.train.task.name == "reg"
        except AttributeError:
            is_reg = False

        # TODO SPARK-LAMA: Fix column names
        train = train_valid_iterator.train.data.withColumn("label", F.lit(0).cast("int")).fillna(0)
        valid = train_valid_iterator.valid.data.withColumn("label", F.lit(0).cast("int")).fillna(0)

        feature_cols = train.columns[1:]
        featurizer = VectorAssembler(
            inputCols=feature_cols,
            outputCol='features'
        )
        train_data = featurizer.transform(train)
        test_data = featurizer.transform(train)

        (
            params,
            num_trees,
            early_stopping_rounds,
            verbose_eval,
            fobj,
            feval,
        ) = self._infer_params()

        LGBMBooster = LightGBMRegressor if is_reg else LightGBMClassifier

        lgbm = LGBMBooster(
            # fobj=fobj,  # TODO SPARK-LAMA: Commented only for smoke test
            # feval=feval,
            learningRate=params["learning_rate"],
            numLeaves=params["num_leaves"],
            featureFraction=params["feature_fraction"],
            baggingFraction=params["bagging_fraction"],
            baggingFreq=params["bagging_freq"],
            maxDepth=params["max_depth"],
            verbosity=params["verbosity"],
            minGainToSplit=params["min_split_gain"],
            numThreads=params["num_threads"],
            maxBin=params["max_bin"],
            minDataInLeaf=params["min_data_in_bin"],
            earlyStoppingRound=early_stopping_rounds
        )
        if is_reg:
            lgbm.setAlpha(params["reg_alpha"]).setLambdaL1(params["reg_lambda"]).setLambdaL2(params["reg_lambda"])


        predicted = lgbm.fit(train_data).transform(test_data)

        predicted.show(10)

        output = train_valid_iterator.train.empty()
        new_roles = train_valid_iterator.valid.roles
        new_roles["label"] = CategoryRole(np.int32)

        cols = train_valid_iterator.valid.data.columns
        cols.extend(["label"])

        output.set_data(data=predicted.select(cols), features=train_valid_iterator.valid.features, roles=new_roles)
        return output


    def fit(self, train_valid: TrainValidIterator):
        """Just to be compatible with :class:`~lightautoml.pipelines.selection.base.ImportanceEstimator`.

        Args:
            train_valid: Classic cv-iterator.

        """
        self.fit_predict(train_valid)
