import functools
import logging
import random
import time
import warnings

from copy import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import lightgbm as lgb
import pandas as pd
import pyspark.sql.functions as sf

from lightautoml.ml_algo.tuning.base import Distribution
from lightautoml.ml_algo.tuning.base import SearchSpace
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.utils.timer import TaskTimer
from lightautoml.validation.base import TrainValidIterator
from lightgbm import Booster
from pandas import Series
from pyspark.ml import PipelineModel
from pyspark.ml import Transformer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.util import MLReadable
from pyspark.ml.util import MLWritable
from pyspark.ml.util import MLWriter
from synapse.ml.lightgbm import LightGBMClassificationModel
from synapse.ml.lightgbm import LightGBMClassifier
from synapse.ml.lightgbm import LightGBMRegressionModel
from synapse.ml.lightgbm import LightGBMRegressor
from synapse.ml.onnx import ONNXModel

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.ml_algo.base import AveragingTransformer
from sparklightautoml.ml_algo.base import ComputationalParameters
from sparklightautoml.ml_algo.base import SparkMLModel
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo
from sparklightautoml.mlwriters import LightGBMModelWrapperMLReader
from sparklightautoml.mlwriters import LightGBMModelWrapperMLWriter
from sparklightautoml.mlwriters import ONNXModelWrapperMLReader
from sparklightautoml.mlwriters import ONNXModelWrapperMLWriter
from sparklightautoml.transformers.base import DropColumnsTransformer
from sparklightautoml.transformers.base import PredictionColsTransformer
from sparklightautoml.transformers.base import ProbabilityColsTransformer
from sparklightautoml.utils import SparkDataFrame
from sparklightautoml.validation.base import SparkBaseTrainValidIterator
from sparklightautoml.validation.base import split_out_val


logger = logging.getLogger(__name__)


class LightGBMModelWrapper(Transformer, MLWritable, MLReadable):
    """Simple wrapper for `synapse.ml.lightgbm.[LightGBMRegressionModel|LightGBMClassificationModel]`
    to fix issue with loading model from saved composite pipeline.

    For more details see: https://github.com/microsoft/SynapseML/issues/614.
    """

    def __init__(self, model: Union[LightGBMRegressionModel, LightGBMClassificationModel] = None) -> None:
        super().__init__()
        self.model = model

    def write(self) -> MLWriter:
        return LightGBMModelWrapperMLWriter(self)

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return LightGBMModelWrapperMLReader()

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        return self.model.transform(dataset)


class ONNXModelWrapper(Transformer, MLWritable, MLReadable):
    """Simple wrapper for `ONNXModel` to fix issue with loading model from saved composite pipeline.

    For more details see: https://github.com/microsoft/SynapseML/issues/614.
    """

    def __init__(self, model: ONNXModel = None) -> None:
        super().__init__()
        self.model = model

    def write(self) -> MLWriter:
        return ONNXModelWrapperMLWriter(self)

    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return ONNXModelWrapperMLReader()

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        return self.model.transform(dataset)


class SparkBoostLGBM(SparkTabularMLAlgo, ImportanceEstimator):
    """Gradient boosting on decision trees from LightGBM library.

    default_params: All available parameters listed in synapse.ml documentation:

    - https://mmlspark.blob.core.windows.net/docs/0.9.5/pyspark/synapse.ml.lightgbm.html#module-synapse.ml.lightgbm.LightGBMClassifier
    - https://mmlspark.blob.core.windows.net/docs/0.9.5/pyspark/synapse.ml.lightgbm.html#module-synapse.ml.lightgbm.LightGBMRegressor

    freeze_defaults:

        - ``True`` :  params may be rewritten depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """

    _name: str = "LightGBM"

    _default_params = {
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
    }

    # mapping between metric name defined via SparkTask
    # and metric names supported by LightGBM
    _metric2lgbm = {
        "binary": {"auc": "auc", "aupr": "areaUnderPR"},
        "reg": {
            "r2": "rmse",
            "mse": "mse",
            "mae": "mae",
        },
        "multiclass": {"crossentropy": "cross_entropy"},
    }

    def __init__(
        self,
        default_params: Optional[dict] = None,
        freeze_defaults: bool = True,
        timer: Optional[TaskTimer] = None,
        optimization_search_space: Optional[dict] = None,
        use_single_dataset_mode: bool = True,
        max_validation_size: int = 10_000,
        chunk_size: int = 4_000_000,
        convert_to_onnx: bool = False,
        mini_batch_size: int = 5000,
        seed: int = 42,
        parallelism: int = 1,
        execution_mode: str = "bulk",
        use_barrier_execution_mode: bool = False,
        experimental_parallel_mode: bool = False,
        persist_output_dataset: bool = True,
        computations_settings: Optional[ComputationalParameters] = None,
    ):
        optimization_search_space = optimization_search_space if optimization_search_space else dict()
        SparkTabularMLAlgo.__init__(
            self,
            default_params,
            freeze_defaults,
            timer,
            optimization_search_space,
            persist_output_dataset,
            computations_settings,
        )
        self._probability_col_name = "probability"
        self._prediction_col_name = "prediction"
        self._raw_prediction_col_name = "raw_prediction"
        self._assembler = None
        self._drop_cols_transformer = None
        self._use_single_dataset_mode = use_single_dataset_mode
        self._max_validation_size = max_validation_size
        self._seed = seed
        self._models_feature_importances = []
        self._chunk_size = chunk_size
        self._convert_to_onnx = convert_to_onnx
        self._mini_batch_size = mini_batch_size
        self._parallelism = parallelism
        self._executin_mode = execution_mode
        self._use_barrier_execution_mode = use_barrier_execution_mode
        self._experimental_parallel_mode = experimental_parallel_mode

    def _infer_params(self, runtime_settings: Optional[Dict[str, Any]] = None) -> Tuple[dict, int]:
        """Infer all parameters in lightgbm format.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval).
            About parameters: https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/engine.html

        """
        assert self.task is not None

        # TODO: PARALLEL - validate runtime settings

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

        runtime_settings = runtime_settings or dict()
        if "num_tasks" in runtime_settings:
            params["numTasks"] = runtime_settings["num_tasks"]
        if "num_threads" in runtime_settings:
            params["numThreads"] = runtime_settings["num_threads"]

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

        Args:.
            suggested_params: suggested params
            estimated_n_trials: Number of trials.

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
            optimization_search_space["isUnbalance"] = SearchSpace(Distribution.DISCRETEUNIFORM, low=0, high=1, q=1)

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

    def predict_single_fold(self, dataset: SparkDataset, model: PipelineModel) -> SparkDataFrame:
        return model.transform(dataset.data)

    def _do_convert_to_onnx(self, train: SparkDataset, ml_model):
        logger.info("Model convert is started")
        booster_model_str = ml_model.getLightGBMBooster().modelStr().get()
        booster = lgb.Booster(model_str=booster_model_str)
        model_payload_ml = self._convert_model(booster, len(train.features))

        onnx_ml = ONNXModel().setModelPayload(model_payload_ml)

        if train.task.name == "reg":
            onnx_ml = (
                onnx_ml.setDeviceType("CPU")
                .setFeedDict({"input": f"{self._name}_vassembler_features"})
                .setFetchDict({ml_model.getPredictionCol(): "variable"})
                .setMiniBatchSize(self._mini_batch_size)
            )
        else:
            onnx_ml = (
                onnx_ml.setDeviceType("CPU")
                .setFeedDict({"input": f"{self._name}_vassembler_features"})
                .setFetchDict({ml_model.getProbabilityCol(): "probabilities", ml_model.getPredictionCol(): "label"})
                .setMiniBatchSize(self._mini_batch_size)
            )

        logger.info("Model convert is ended")

        return onnx_ml

    def fit_predict_single_fold(
        self,
        fold_prediction_column: str,
        validation_column: str,
        train: SparkDataset,
        runtime_settings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[SparkMLModel, SparkDataFrame, str]:
        if self.task is None:
            self.task = train.task

        (params, verbose_eval) = self._infer_params(runtime_settings)

        logger.info(f"Input cols for the vector assembler: {train.features}")
        logger.info(f"Running lgb with the following params: {params}")

        if train.task.name in ["binary", "multiclass"]:
            params["rawPredictionCol"] = self._raw_prediction_col_name
            params["probabilityCol"] = fold_prediction_column
            params["predictionCol"] = self._prediction_col_name
            params["isUnbalance"] = True
        else:
            params["predictionCol"] = fold_prediction_column

        assert (
            validation_column in train.data.columns
        ), f"Validation column {validation_column} should be present in the data"

        full_data = self._ensure_validation_size(train.data, validation_column)

        # prepare assembler
        if self._assembler is None:
            self._assembler = VectorAssembler(
                inputCols=train.features, outputCol=f"{self._name}_vassembler_features", handleInvalid="keep"
            )

        # assign a random port to decrease chances of allocating the same port from multiple instances
        rand = random.Random(time.time_ns())
        random_port = rand.randint(10_000, 50_000)

        run_params = {
            "featuresCol": self._assembler.getOutputCol(),
            "labelCol": train.target_column,
            "validationIndicatorCol": validation_column,
            "verbosity": verbose_eval,
            "executionMode": self._executin_mode,
            "useSingleDatasetMode": self._use_single_dataset_mode,
            "useBarrierExecutionMode": self._use_barrier_execution_mode,
            "isProvideTrainingMetric": True,
            "chunkSize": self._chunk_size,
            "defaultListenPort": random_port,
            **params,
            **({"alpha": 0.5, "lambdaL1": 0.0, "lambdaL2": 0.0} if train.task.name == "reg" else dict()),
        }

        # build the booster
        lgbm_booster = LightGBMRegressor if train.task.name == "reg" else LightGBMClassifier
        lgbm = lgbm_booster(**run_params)

        logger.info(f"Use single dataset mode: {lgbm.getUseSingleDatasetMode()}. NumThreads: {lgbm.getNumThreads()}")
        logger.info(f"All lgbm booster params: {run_params}")

        rows_count = full_data.count()
        if (run_params["executionMode"] == "streaming") and (rows_count <= 25_000):
            warnings.warn(
                f"The fitting of lightgbm in streaming execution mode "
                f"may fail with SEGSIGV / SIGBUS error (probably due to a bug in synapse ml) "
                f"if too few data available per core. Train data rows count: {rows_count} "
                f"Consider switching to bulk execution mode if such crashes happen",
                RuntimeWarning,
            )

        # fitting the model
        ml_model = lgbm.fit(self._assembler.transform(full_data))

        # handle the model
        ml_model = self._do_convert_to_onnx(train, ml_model) if self._convert_to_onnx else ml_model
        self._models_feature_importances.append(ml_model.getFeatureImportances(importance_type="gain"))

        valid_data = split_out_val(full_data, validation_column)
        # predict validation
        val_pred = ml_model.transform(self._assembler.transform(valid_data))
        val_pred = DropColumnsTransformer(
            remove_cols=[],
            optional_remove_cols=[self._prediction_col_name, self._probability_col_name, self._raw_prediction_col_name],
        ).transform(val_pred)

        return ml_model, val_pred, fold_prediction_column

    def fit(self, train_valid: SparkBaseTrainValidIterator):
        logger.info("Starting LGBM fit")
        self.fit_predict(train_valid)
        logger.info("Finished LGBM fit")

    def get_features_score(self) -> Series:
        imp = functools.reduce(lambda acc, x: acc + pd.Series(x), self._models_feature_importances, 0)

        # imp = 0
        # for model_feature_impotances in self._models_feature_importances:
        #     imp += pd.Series(model_feature_impotances)

        imp /= len(self._models_feature_importances)

        def flatten_features(feat: str):
            role = self.input_roles[feat]
            if isinstance(role, NumericVectorOrArrayRole):
                return [f"{feat}_pos_{i}" for i in range(role.size)]
            return [feat]

        index = [ff for feat in self._assembler.getInputCols() for ff in flatten_features(feat)]

        result = Series(list(imp), index=index).sort_values(ascending=False)
        return result

    @staticmethod
    def _convert_model(lgbm_model: Booster, input_size: int) -> bytes:
        from onnxconverter_common.data_types import FloatTensorType
        from onnxmltools.convert import convert_lightgbm

        initial_types = [("input", FloatTensorType([-1, input_size]))]
        onnx_model = convert_lightgbm(lgbm_model, initial_types=initial_types, target_opset=9)
        return onnx_model.SerializeToString()

    def _ensure_validation_size(self, full_data: SparkDataFrame, validation_column: str) -> SparkDataFrame:
        # reduce validation size if it is too big
        val_data_size = full_data.where(sf.col(validation_column).astype("int") == 1).count()
        if val_data_size > self._max_validation_size:
            logger.warning(
                f"Too big validation fold: {val_data_size}. "
                f"Reducing its size down according to max_validation_size setting:"
                f" {self._max_validation_size}"
            )
            not_val_row = sf.col(validation_column) != sf.lit(1)
            not_chosen_as_val = sf.rand(seed=self._seed) < sf.lit(self._max_validation_size / val_data_size)
            full_data = full_data.where(not_val_row | not_chosen_as_val)

        # checking if there are no empty partitions that may lead to hanging
        rows = (
            full_data.withColumn("__partition_id__", sf.spark_partition_id())
            .groupby("__partition_id__")
            .agg(sf.sum(validation_column).alias("val_values"), sf.count("*").alias("all_values"))
            .collect()
        )
        for row in rows:
            if row["val_values"] == row["all_values"] or row["all_values"] == 0:
                warnings.warn(
                    f"Empty partition encountered: partition id - {row['__partition_id_']},"
                    f"validation values count in the partition - {row['val_values']}, "
                    f"all values count in the partition  - {row['all_values']}"
                )
                raise ValueError(
                    f"Empty partition encountered: partition id - {row['__partition_id_']},"
                    f"validation values count in the partition - {row['val_values']}, "
                    f"all values count in the partition  - {row['all_values']}"
                )

        return full_data

    def _build_transformer(self) -> Transformer:
        avr = self._build_averaging_transformer()
        if self._convert_to_onnx:
            wrapped_models = [ONNXModelWrapper(m) for m in self.models]
        else:
            wrapped_models = [LightGBMModelWrapper(m) for m in self.models]
        models: List[Transformer] = [
            el
            for m in wrapped_models
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
        if self._convert_to_onnx:
            if self.task.name in ["binary", "multiclass"]:
                models.append(
                    ProbabilityColsTransformer(
                        probability_cols=self._models_prediction_columns, num_classes=self.n_classes
                    )
                )
            else:
                models.append(PredictionColsTransformer(prediction_cols=self._models_prediction_columns))
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
        logger.info("Starting LGBM fit")
        self.timer.start()

        res = super().fit_predict(train_valid_iterator)

        logger.info("Finished LGBM fit")
        return res
