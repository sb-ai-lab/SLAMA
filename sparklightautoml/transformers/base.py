import logging

from abc import ABC
from copy import copy
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import cast

from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import ColumnRole
from lightautoml.transformers.base import LAMLTransformer
from pyspark.ml import Estimator
from pyspark.ml import Transformer
from pyspark.ml.functions import array_to_vector
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.param.shared import HasOutputCols
from pyspark.ml.param.shared import Param
from pyspark.ml.param.shared import Params
from pyspark.ml.param.shared import TypeConverters
from pyspark.ml.util import DefaultParamsReadable
from pyspark.ml.util import DefaultParamsWritable
from pyspark.sql import Column
from pyspark.sql import functions as sf

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.mlwriters import CommonPickleMLReadable
from sparklightautoml.mlwriters import CommonPickleMLWritable
from sparklightautoml.utils import SparkDataFrame


logger = logging.getLogger(__name__)

SparkEstOrTrans = Union[
    "SparkBaseEstimator", "SparkBaseTransformer", "SparkUnionTransformer", "SparkSequentialTransformer"
]


class HasInputRoles(Params):
    """
    Mixin for param inputCols: input column names.
    """

    inputRoles = Param(Params._dummy(), "inputRoles", "input roles (lama format)")

    def __init__(self):
        super().__init__()

    def get_input_roles(self):
        """
        Gets the value of inputCols or its default value.
        """
        return self.getOrDefault(self.inputRoles)


class HasOutputRoles(Params):
    """
    Mixin for param inputCols: input column names.
    """

    outputRoles = Param(Params._dummy(), "outputRoles", "output roles (lama format)")

    def __init__(self):
        super().__init__()

    def get_output_roles(self):
        """
        Gets the value of inputCols or its default value.
        """
        return self.getOrDefault(self.outputRoles)


class SparkColumnsAndRoles(HasInputCols, HasOutputCols, HasInputRoles, HasOutputRoles):
    """
    Helper and base class for :class:`~sparklightautoml.transformers.base.SparkBaseTransformer` and
    :class:`~sparklightautoml.transformers.base.SparkBaseEstimator`.
    """

    _fname_prefix = ""

    doReplaceColumns = Param(Params._dummy(), "doReplaceColumns", "whatever it replaces columns or not")
    columnsToReplace = Param(Params._dummy(), "columnsToReplace", "which columns to replace")

    def get_do_replace_columns(self) -> bool:
        return self.getOrDefault(self.doReplaceColumns)

    def get_columns_to_replace(self) -> List[str]:
        return self.getOrDefault(self.columnsToReplace)

    @staticmethod
    def make_dataset(
        transformer: "SparkColumnsAndRoles", base_dataset: SparkDataset, data: SparkDataFrame
    ) -> SparkDataset:
        new_roles = deepcopy(base_dataset.roles)
        new_roles.update(transformer.get_output_roles())
        new_ds = base_dataset.empty()
        new_ds.set_data(data, base_dataset.features + transformer.getOutputCols(), new_roles, name=base_dataset.name)
        return new_ds

    def get_prefix(self):
        return self._fname_prefix


class SparkBaseEstimator(Estimator, SparkColumnsAndRoles, ABC):
    """
    Base class for estimators from `sparklightautoml.transformers`.
    """

    _fit_checks = ()
    _fname_prefix = ""

    def __init__(
        self,
        input_cols: Optional[List[str]] = None,
        input_roles: Optional[Dict[str, ColumnRole]] = None,
        do_replace_columns: bool = False,
        output_role: Optional[ColumnRole] = None,
    ):
        super().__init__()

        self._output_role = output_role

        assert all((f in input_roles) for f in input_cols), "All columns should have roles"

        self.set(self.inputCols, input_cols)
        self.set(self.outputCols, self._make_output_names(input_cols))
        self.set(self.inputRoles, input_roles)
        self.set(self.outputRoles, self._make_output_roles())
        self.set(self.doReplaceColumns, do_replace_columns)

    def _make_output_names(self, input_cols: List[str]) -> List[str]:
        return [f"{self._fname_prefix}__{feat}" for feat in input_cols]

    def _make_output_roles(self):
        assert self._output_role is not None
        new_roles = {}
        new_roles.update({feat: self._output_role for feat in self.getOutputCols()})
        return new_roles


class SparkBaseTransformer(Transformer, SparkColumnsAndRoles, ABC):
    """
    Base class for transformers from `sparklightautoml.transformers`.
    """

    _fname_prefix = ""

    def __init__(
        self,
        input_cols: List[str],
        output_cols: List[str],
        input_roles: RolesDict,
        output_roles: RolesDict,
        do_replace_columns: Union[bool, List[str]] = False,
    ):
        super(SparkBaseTransformer, self).__init__()

        # assert len(input_cols) == len(output_cols)
        # assert len(input_roles) == len(output_roles)
        assert all((f in input_roles) for f in input_cols)
        assert all((f in output_roles) for f in output_cols)

        self.set(self.inputCols, input_cols)
        self.set(self.outputCols, output_cols)
        self.set(self.inputRoles, input_roles)
        self.set(self.outputRoles, output_roles)

        if isinstance(do_replace_columns, List):
            cols_to_replace = cast(List[str], do_replace_columns)
            assert (
                len(set(cols_to_replace).difference(set(self.getInputCols()))) == 0
            ), "All columns to replace, should be in input columns"
            self.set(self.doReplaceColumns, True)
            self.set(self.columnsToReplace, cols_to_replace)
        else:
            self.set(self.doReplaceColumns, do_replace_columns)
            self.set(self.columnsToReplace, self.getInputCols() if do_replace_columns else [])

    _transform_checks = ()

    # noinspection PyMethodMayBeStatic
    def _make_output_df(self, input_df: SparkDataFrame, cols_to_add: List[Union[str, Column]]):
        return input_df.select("*", *cols_to_add)

        # if not self.getDoReplaceColumns():
        #     return input_df.select('*', *cols_to_add)
        #
        # replaced_columns = set(self.getColumnsToReplace())
        # cols_to_leave = [f for f in input_df.columns if f not in replaced_columns]
        # return input_df.select(*cols_to_leave, *cols_to_add)

    def transform(self, dataset, params=None):
        logger.debug(f"Transforming {type(self)}. Columns: {sorted(dataset.columns)}")
        transformed_dataset = super().transform(dataset, params)
        logger.debug(f"Out {type(self)}. Columns: {sorted(transformed_dataset.columns)}")
        return transformed_dataset


class SparkUnionTransformer:
    """
    Entity that represents parallel layers (transformers) in preprocess pipeline.
    """

    def __init__(self, transformer_list: List[SparkEstOrTrans]):
        self._transformer_list = copy(transformer_list)

    @property
    def transformers(self) -> List[SparkEstOrTrans]:
        return self._transformer_list

    # noinspection PyMethodMayBeStatic
    def _find_last_stage(self, stage):
        if isinstance(stage, SparkSequentialTransformer):
            stage = stage.transformers[-1]
            return stage
        return stage

    def get_output_cols(self) -> List[str]:
        """Get list of output columns from all stages

        Returns:
            List[str]: output columns
        """
        output_cols = []
        for stage in self._transformer_list:
            stage = self._find_last_stage(stage)
            output_cols.extend(stage.getOutputCols())
        return list(set(output_cols))

    def get_output_roles(self) -> RolesDict:
        """Get output roles from all stages

        Returns:
            RolesDict: output roles
        """
        roles = {}
        for stage in self._transformer_list:
            stage = self._find_last_stage(stage)
            roles.update(deepcopy(stage.get_output_roles()))

        return roles


class SparkSequentialTransformer:
    """
    Entity that represents sequential of transformers in preprocess pipeline.
    """

    def __init__(self, transformer_list: List[SparkEstOrTrans]):
        self._transformer_list = copy(transformer_list)

    @property
    def transformers(self) -> List[SparkEstOrTrans]:
        return self._transformer_list


class ObsoleteSparkTransformer(LAMLTransformer):
    _features = []

    _can_unwind_parents: bool = True

    _input_features = []

    def get_output_names(self, input_cols: List[str]) -> List[str]:
        return [f"{self._fname_prefix}__{feat}" for feat in input_cols]

    def fit(self, dataset: SparkDataset, use_features: Optional[List[str]] = None) -> "ObsoleteSparkTransformer":
        logger.info(f"SparkTransformer of type: {type(self)}")

        if use_features:
            existing_feats = set(dataset.features)
            not_found_feats = [feat for feat in use_features if feat not in existing_feats]
            assert len(not_found_feats) == 0, f"Not found features {not_found_feats} among existing {existing_feats}"
            self._features = use_features
        else:
            self._features = dataset.features

        self._input_features = use_features if use_features else dataset.features

        for check_func in self._fit_checks:
            check_func(dataset, use_features)

        return self._fit(dataset)

    def _fit(self, dataset: SparkDataset) -> "ObsoleteSparkTransformer":
        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        for check_func in self._transform_checks:
            check_func(dataset)

        return self._transform(dataset)

    def _transform(self, dataset: SparkDataset) -> SparkDataset:
        return dataset

    def fit_transform(self, dataset: SparkDataset) -> SparkDataset:
        logger.info(f"fit_transform in {self._fname_prefix}: {type(self)}")

        self.fit(dataset)

        result = self.transform(dataset)

        return result

    @staticmethod
    def _get_updated_roles(dataset: SparkDataset, new_features: List[str], new_role: ColumnRole) -> RolesDict:
        new_roles = deepcopy(dataset.roles)
        new_roles.update({feat: new_role for feat in new_features})
        return new_roles


class ProbabilityColsTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    """Converts probability columns values from ONNX model format to LGBMCBooster format"""

    probabilityCols = Param(
        Params._dummy(),
        "probabilityCols",
        "probability сolumn names in dataframe",
        typeConverter=TypeConverters.toListString,
    )

    numClasses = Param(Params._dummy(), "numClasses", "number of classes", typeConverter=TypeConverters.toInt)

    def __init__(self, probability_cols: Optional[List[str]] = None, num_classes: int = 0):
        super().__init__()
        probability_cols = probability_cols if probability_cols else []
        self.set(self.probabilityCols, probability_cols)
        self.set(self.numClasses, num_classes)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        logger.debug(f"In {type(self)}. Columns: {sorted(dataset.columns)}")
        num_classes = self.getOrDefault(self.numClasses)
        probability_cols = self.getOrDefault(self.probabilityCols)
        other_cols = [c for c in dataset.columns if c not in probability_cols]
        probability_cols = [
            array_to_vector(sf.array(*[sf.col(c).getItem(i) for i in range(num_classes)])).alias(c)
            for c in probability_cols
        ]
        dataset = dataset.select(*other_cols, *probability_cols)
        logger.debug(f"Out {type(self)}. Columns: {sorted(dataset.columns)}")
        return dataset


class PredictionColsTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    """Converts prediction columns values from ONNX model format to LGBMCBooster format"""

    predictionCols = Param(
        Params._dummy(),
        "predictionCols",
        "probability сolumn names in dataframe",
        typeConverter=TypeConverters.toListString,
    )

    def __init__(self, prediction_cols: Optional[List[str]] = None):
        super().__init__()
        prediction_cols = prediction_cols if prediction_cols else []
        self.set(self.predictionCols, prediction_cols)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        logger.debug(f"In {type(self)}. Columns: {sorted(dataset.columns)}")
        prediction_cols = self.getOrDefault(self.predictionCols)
        other_cols = [c for c in dataset.columns if c not in prediction_cols]
        prediction_cols = [sf.col(c).getItem(0).alias(c) for c in prediction_cols]
        dataset = dataset.select(*other_cols, *prediction_cols)
        logger.debug(f"Out {type(self)}. Columns: {sorted(dataset.columns)}")
        return dataset


class DropColumnsTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    """Transformer that drops columns from input dataframe."""

    colsToRemove = Param(
        Params._dummy(), "colsToRemove", "columns to be removed", typeConverter=TypeConverters.toListString
    )

    optionalColsToRemove = Param(
        Params._dummy(),
        "optionalColsToRemove",
        "optional columns to be removed (if they appear in the dataset)",
        typeConverter=TypeConverters.toListString,
    )

    def __init__(self, remove_cols: Optional[List[str]] = None, optional_remove_cols: Optional[List[str]] = None):
        super().__init__()
        remove_cols = remove_cols if remove_cols else []
        self.set(self.colsToRemove, remove_cols)
        self.set(self.optionalColsToRemove, optional_remove_cols if optional_remove_cols else [])

    def get_cols_to_remove(self) -> List[str]:
        return self.getOrDefault(self.colsToRemove)

    def get_optional_cols_to_remove(self) -> List[str]:
        return self.getOrDefault(self.optionalColsToRemove)

    def _transform(self, dataset):
        logger.debug(f"In {type(self)}. Columns: {sorted(dataset.columns)}")

        ds_cols = set(dataset.columns)
        optional_to_remove = [c for c in self.get_optional_cols_to_remove() if c in ds_cols]
        dataset = dataset.drop(*self.get_cols_to_remove(), *optional_to_remove)

        logger.debug(f"Out {type(self)}. Columns: {sorted(dataset.columns)}")
        return dataset


class SparkChangeRolesTransformer(SparkBaseTransformer, CommonPickleMLWritable, CommonPickleMLReadable):
    """Transformer that change roles for input columns. Does not transform the input dataframe.

    # Note: this trasnformer cannot be applied directly to input columns of a feature pipeline
    """

    def __init__(
        self,
        input_cols: List[str],
        input_roles: RolesDict,
        role: Optional[ColumnRole] = None,
        roles: Optional[RolesDict] = None,
    ):
        assert (role and not roles) or (
            not role and roles
        ), "Either role or roles shoud be defined. Both of them cannot be set simultaneously"

        if role:
            output_roles = {f: deepcopy(role) for f in input_cols}
        else:
            output_roles = deepcopy(roles)

        super().__init__(
            input_cols=input_cols,
            output_cols=input_cols,
            input_roles=input_roles,
            output_roles=output_roles,
            do_replace_columns=True,
        )

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        return dataset
