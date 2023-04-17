import logging
import os
from copy import copy
from typing import List, cast, Optional, Any, Tuple, Callable

import numpy as np
import pyspark.sql.functions as sf
from lightautoml.dataset.roles import NumericRole
from lightautoml.reader.base import UserDefinedRolesDict
from lightautoml.reader.tabular_batch_generator import ReadableToDf
from pyspark.ml import Transformer, PipelineModel
from pyspark.ml.feature import VectorSizeHint
from pyspark.ml.functions import array_to_vector

from sparklightautoml.automl.blend import SparkWeightedBlender
from sparklightautoml.automl.presets.base import SparkAutoMLPreset
from sparklightautoml.dataset.base import SparkDataset, PersistenceManager, PersistenceLevel
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo, SparkMLModel, AveragingTransformer
from sparklightautoml.mlwriters import CommonPickleMLWritable, CommonPickleMLReadable
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import SparkDataFrame
from sparklightautoml.validation.base import SparkBaseTrainValidIterator, split_out_val
from sparklightautoml.validation.iterators import SparkFoldsIterator

logger = logging.getLogger(__name__)


class FakeOpTransformer(Transformer, CommonPickleMLWritable, CommonPickleMLReadable):
    def __init__(self, cols_to_generate: List[str], n_classes: int):
        super().__init__()
        self._cos_to_generate = cols_to_generate
        self._n_classes = n_classes

    def _transform(self, dataset):
        logger.warning(f"In {type(self)}. Columns: {sorted(dataset.columns)}")
        out_dataset = dataset.select(
            '*',
            *[
                array_to_vector(sf.array(*[sf.rand() for _ in range(self._n_classes)])).alias(f)
                for f in self._cos_to_generate
            ]
        )
        logger.warning(f"Out {type(self)}. Columns: {sorted(out_dataset.columns)}")
        return out_dataset


class DummyReader(SparkToSparkReader):
    def __init__(self, task: SparkTask):
        super().__init__(task)

    def fit_read(self,
                 train_data: SparkDataFrame,
                 features_names: Any = None,
                 roles: UserDefinedRolesDict = None,
                 persistence_manager: Optional[PersistenceManager] = None,
                 **kwargs: Any) -> SparkDataset:

        self.target_col = roles["target"]
        self._roles = {c: NumericRole() for c in train_data.columns if c != self.target_col}
        self._used_features = list(self._roles.keys())

        train_data = self._create_unique_ids(train_data)
        train_data, folds_col = self._create_folds(train_data, kwargs={})

        sds = SparkDataset(
            train_data,
            self._roles,
            persistence_manager,
            task=self.task,
            target=self.target_col,
            folds=folds_col,
            name="DummySparkToSparkReader"
        )

        sds = persistence_manager.persist(sds, level=PersistenceLevel.READER).to_dataset()

        return sds

    def read(self, data: SparkDataFrame, features_names: Any = None, add_array_attrs: bool = False) -> SparkDataset:
        data = self._create_unique_ids(data)
        sds = SparkDataset(
            data,
            self._roles,
            persistence_manager=PlainCachePersistenceManager(),
            task=self.task,
            target=self.target_col
        )
        return sds


class DummyMLAlgo(SparkTabularMLAlgo):

    def __init__(self, n_classes: int, name: str):
        self._name = name
        super().__init__()
        self.n_classes = n_classes

    def fit_predict_single_fold(self, fold_prediction_column: str, validation_column: str, train: SparkDataset) \
            -> Tuple[SparkMLModel, SparkDataFrame, str]:
        fake_op = FakeOpTransformer(cols_to_generate=[fold_prediction_column], n_classes=self.n_classes)
        ml_model = PipelineModel(stages=[fake_op])

        valid_data = split_out_val(train.data, validation_column)

        return ml_model, ml_model.transform(valid_data), fold_prediction_column

    def predict_single_fold(self, model: SparkMLModel, dataset: SparkDataset) -> SparkDataFrame:
        raise NotImplementedError("")

    def _build_transformer(self) -> Transformer:
        avr = self._build_averaging_transformer()
        averaging_model = PipelineModel(stages=self.models + [avr])
        return averaging_model

    def _build_averaging_transformer(self) -> Transformer:
        avr = AveragingTransformer(
            self.task.name,
            input_cols=self._models_prediction_columns,
            output_col=self.prediction_feature,
            remove_cols=self._models_prediction_columns,
            convert_to_array_first=not (self.task.name == "reg"),
            dim_num=self.n_classes
        )
        return avr


class DummySparkMLPipeline(SparkMLPipeline):
    def __init__(
        self,
        name: str = "dummy_pipe"
    ):
        super().__init__([], force_calc=[True], name=name)

    def fit_predict(self, train_valid: SparkBaseTrainValidIterator) -> SparkDataset:
        val_ds = train_valid.get_validation_data()

        n_classes = 10

        self._output_roles = {
            f"predictions_{self.name}_alg_{i}":
                NumericVectorOrArrayRole(size=n_classes,
                                         element_col_name_template=f"{self.name}_alg_{i}" + "_{}",
                                         dtype=np.float32,
                                         force_input=True,
                                         prob=False)
            for i in range(3)
        }

        self._transformer = FakeOpTransformer(cols_to_generate=list(self.output_roles.keys()), n_classes=n_classes)

        sdf = cast(SparkDataFrame, val_ds.data)
        sdf = sdf.select(
            '*',
            *[
                array_to_vector(sf.array(*[sf.lit(i * 10 + j) for j in range(n_classes)])).alias(name)
                for i, name in enumerate(self._output_roles.keys())
            ]
        )

        vshs = PipelineModel(stages=[
            VectorSizeHint(inputCol=name, size=n_classes) for name in self._output_roles.keys()
        ])
        sdf = vshs.transform(sdf)

        out_roles = copy(self._output_roles)
        out_roles.update(train_valid.train.roles)
        out_roles.update(train_valid.input_roles)

        out_val_ds = cast(SparkDataset, val_ds.empty())
        out_val_ds.set_data(sdf, list(out_roles.keys()), out_roles, name=val_ds.name)

        return out_val_ds


class DummyTabularAutoML(SparkAutoMLPreset):
    def __init__(self, n_classes: int):
        config_path = os.path.join(os.getcwd(), 'sparklightautoml/automl/presets/tabular_config.yml')
        super().__init__(SparkTask("multiclass"), config_path=config_path)
        self._n_classes = n_classes

    def _create_validation_iterator(self, train: SparkDataset, valid: Optional[SparkDataset], n_folds: Optional[int],
                                    cv_iter: Optional[Callable]) -> SparkBaseTrainValidIterator:
        return SparkFoldsIterator(train, n_folds)

    def create_automl(self, **fit_args):
        # initialize
        reader = DummyReader(self.task)

        first_level = [
            SparkMLPipeline(ml_algos=[DummyMLAlgo(self._n_classes, name=f"dummy_0_{i}")])
            for i in range(3)
        ]
        second_level = [
            SparkMLPipeline(ml_algos=[DummyMLAlgo(self._n_classes, name=f"dummy_1_{i}")])
            for i in range(2)
        ]

        levels = [first_level, second_level]

        blender = SparkWeightedBlender(max_iters=0, max_inner_iters=1)

        self._initialize(
            reader,
            levels,
            skip_conn=True,
            blender=blender,
            return_all_predictions=False,
            timer=self.timer,
        )

    def get_individual_pdp(self, test_data: ReadableToDf, feature_name: str, n_bins: Optional[int] = 30,
                           top_n_categories: Optional[int] = 10, datetime_level: Optional[str] = "year"):
        raise ValueError("Not supported")

    def plot_pdp(self, test_data: ReadableToDf, feature_name: str, individual: Optional[bool] = False,
                 n_bins: Optional[int] = 30, top_n_categories: Optional[int] = 10, top_n_classes: Optional[int] = 10,
                 datetime_level: Optional[str] = "year"):
        raise ValueError("Not supported")
