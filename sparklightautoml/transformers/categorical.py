import itertools
import logging

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import pandas as pd

from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import ColumnRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.reader.base import RolesDict
from lightautoml.transformers.categorical import categorical_check
from lightautoml.transformers.categorical import encoding_check
from lightautoml.transformers.categorical import multiclass_task_check
from lightautoml.transformers.categorical import oof_task_check
from pyspark.ml import Transformer
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql import Column
from pyspark.sql import Window
from pyspark.sql import functions as sf
from pyspark.sql.types import IntegerType
from sklearn.utils.murmurhash import murmurhash3_32

from sparklightautoml.dataset.roles import NumericVectorOrArrayRole
from sparklightautoml.mlwriters import CommonPickleMLReadable
from sparklightautoml.mlwriters import CommonPickleMLWritable
from sparklightautoml.mlwriters import SparkLabelEncoderTransformerMLReadable
from sparklightautoml.mlwriters import SparkLabelEncoderTransformerMLWritable
from sparklightautoml.transformers.base import SparkBaseEstimator
from sparklightautoml.transformers.base import SparkBaseTransformer
from sparklightautoml.transformers.scala_wrappers.laml_string_indexer import (
    LAMLStringIndexer,
)
from sparklightautoml.transformers.scala_wrappers.laml_string_indexer import (
    LAMLStringIndexerModel,
)
from sparklightautoml.transformers.scala_wrappers.target_encoder_transformer import (
    SparkTargetEncodeTransformer,
)
from sparklightautoml.transformers.scala_wrappers.target_encoder_transformer import (
    TargetEncoderTransformer,
)
from sparklightautoml.utils import SparkDataFrame


logger = logging.getLogger(__name__)


# FIXME SPARK-LAMA: np.nan in str representation is 'nan' while Spark's NaN is 'NaN'. It leads to different hashes.
# FIXME SPARK-LAMA: If udf is defined inside the class, it not works properly.
# "if murmurhash3_32 can be applied to a whole pandas Series, it would be better to make it via pandas_udf"
# https://github.com/fonhorst/LightAutoML/pull/57/files/57c15690d66fbd96f3ee838500de96c4637d59fe#r749534669
murmurhash3_32_udf = sf.udf(
    lambda value: murmurhash3_32(value.replace("NaN", "nan"), seed=42) if value is not None else None,
    IntegerType(),
)


@dataclass
class OOfFeatsMapping:
    folds_column: str
    # count of different categories in the categorical column being processed
    dim_size: int
    # mapping from category to a continious reperesentation found by target encoder
    # category may be represented:
    # - cat (plain category)
    # - dim_size * folds_num + cat
    # mapping: Dict[int, float]
    mapping: np.array


class TypesHelper:
    _ad_hoc_types_mapper = defaultdict(
        lambda: "string",
        {
            "bool": "boolean",
            "int": "int",
            "int8": "int",
            "int16": "int",
            "int32": "int",
            "int64": "int",
            "int128": "bigint",
            "int256": "bigint",
            "integer": "int",
            "uint8": "int",
            "uint16": "int",
            "uint32": "int",
            "uint64": "int",
            "uint128": "bigint",
            "uint256": "bigint",
            "longlong": "long",
            "ulonglong": "long",
            "float16": "float",
            "float": "float",
            "float32": "float",
            "float64": "double",
            "float128": "double",
        },
    )

    _spark_numeric_types_str = ("ShortType", "IntegerType", "LongType", "FloatType", "DoubleType", "DecimalType")


class SparkLabelEncoderEstimator(SparkBaseEstimator, TypesHelper):
    """
    Spark label encoder estimator.
    Returns :class:`~sparklightautoml.transformers.categorical.SparkLabelEncoderTransformer`.
    """

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "le"

    _fillna_val = 0.0

    def __init__(
        self,
        input_cols: Optional[List[str]] = None,
        input_roles: Optional[Dict[str, ColumnRole]] = None,
        subs: Optional[float] = None,
        random_state: Optional[int] = 42,
        do_replace_columns: bool = False,
        output_role: Optional[ColumnRole] = None,
    ):
        if not output_role:
            output_role = CategoryRole(np.int32, label_encoded=True)
        super().__init__(input_cols, input_roles, do_replace_columns=do_replace_columns, output_role=output_role)
        self._input_intermediate_columns = self.getInputCols()
        self._input_intermediate_roles = self.get_input_roles()

    def _fit(self, dataset: SparkDataFrame) -> "SparkLabelEncoderTransformer":
        logger.info(f"[{type(self)} (LE)] fit is started")

        roles = self._input_intermediate_roles
        columns = self._input_intermediate_columns

        # if self._fname_prefix == "inter":
        #     roles = self.get_input_roles()
        #     columns = self.getInputCols()

        indexer = LAMLStringIndexer(
            inputCols=columns,
            outputCols=self.getOutputCols(),
            minFreqs=[roles[col_name].unknown for col_name in columns],
            handleInvalid="keep",
            defaultValue=self._fillna_val,
        )

        self.indexer_model = indexer.fit(dataset)

        logger.info(f"[{type(self)} (LE)] fit is finished")

        return SparkLabelEncoderTransformer(
            input_cols=self.getInputCols(),
            output_cols=self.getOutputCols(),
            input_roles=self.get_input_roles(),
            output_roles=self.get_output_roles(),
            do_replace_columns=self.get_do_replace_columns(),
            indexer_model=self.indexer_model,
        )


class SparkLabelEncoderTransformer(
    SparkBaseTransformer, TypesHelper, SparkLabelEncoderTransformerMLWritable, SparkLabelEncoderTransformerMLReadable
):
    """
    Simple Spark version of `LabelEncoder`.

    Labels are integers from 1 to n.
    """

    _transform_checks = ()
    _fname_prefix = "le"

    _fillna_val = 0.0

    def __init__(
        self,
        input_cols: List[str],
        output_cols: List[str],
        input_roles: RolesDict,
        output_roles: RolesDict,
        do_replace_columns: bool,
        indexer_model: LAMLStringIndexerModel,
    ):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)
        self.indexer_model = indexer_model
        self._input_intermediate_columns = self.getInputCols()
        self._input_intermediate_roles = self.get_input_roles()

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        logger.info(f"[{type(self)} (LE)] transform is started")

        columns = self._input_intermediate_columns
        out_columns = self.getOutputCols()

        model: LAMLStringIndexerModel = (
            self.indexer_model.setDefaultValue(self._fillna_val)
            .setHandleInvalid("keep")
            .setInputCols(columns)
            .setOutputCols(out_columns)
        )

        logger.info(f"[{type(self)} (LE)] Transform is finished")

        output = model.transform(dataset)

        rest_cols = [col for col in output.columns if col not in self.getOutputCols()]
        out_cols = [sf.col(col).astype(IntegerType()).alias(col) for col in self.getOutputCols()]

        output = output.select([*rest_cols, *out_cols])

        return output


class SparkOrdinalEncoderEstimator(SparkLabelEncoderEstimator):
    """
    Spark ordinal encoder estimator.
    Returns :class:`~sparklightautoml.transformers.categorical.SparkOrdinalEncoderTransformer`.
    """

    _fit_checks = (categorical_check,)
    _fname_prefix = "ord"
    _fillna_val = float("nan")

    def __init__(
        self,
        input_cols: Optional[List[str]] = None,
        input_roles: Optional[Dict[str, ColumnRole]] = None,
        subs: Optional[float] = None,
        random_state: Optional[int] = 42,
    ):
        super().__init__(input_cols, input_roles, subs, random_state, output_role=NumericRole(np.float32))
        self.dicts = None
        self._use_cols = self.getInputCols()

    def _fit(self, dataset: SparkDataFrame) -> "Transformer":
        logger.info(f"[{type(self)} (ORD)] fit is started")

        cols_to_process = [
            col for col in self.getInputCols() if str(dataset.schema[col].dataType) not in self._spark_numeric_types_str
        ]

        min_freqs = [self._input_intermediate_roles[col].unknown for col in cols_to_process]

        indexer = LAMLStringIndexer(
            stringOrderType="alphabetAsc",
            inputCols=cols_to_process,
            outputCols=[f"{self._fname_prefix}__{col}" for col in cols_to_process],
            minFreqs=min_freqs,
            handleInvalid="keep",
            defaultValue=self._fillna_val,
            nanLast=True,  # Only for ORD
        )

        self.indexer_model = indexer.fit(dataset)

        logger.info(f"[{type(self)} (ORD)] fit is finished")

        return SparkOrdinalEncoderTransformer(
            input_cols=self.getInputCols(),
            output_cols=self.getOutputCols(),
            input_roles=self.get_input_roles(),
            output_roles=self.get_output_roles(),
            do_replace_columns=self.get_do_replace_columns(),
            indexer_model=self.indexer_model,
        )


class SparkOrdinalEncoderTransformer(SparkLabelEncoderTransformer):
    """
    Spark version of :class:`~lightautoml.transformers.categorical.OrdinalEncoder`.

    Encoding ordinal categories into numbers.
    Number type categories passed as is,
    object type sorted in ascending lexicographical order.
    """

    _transform_checks = ()
    _fname_prefix = "ord"
    _fillna_val = float("nan")

    def __init__(
        self,
        input_cols: List[str],
        output_cols: List[str],
        input_roles: RolesDict,
        output_roles: RolesDict,
        do_replace_columns: bool,
        indexer_model: LAMLStringIndexerModel,
    ):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns, indexer_model)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        logger.info(f"[{type(self)} (ORD)] transform is started")

        input_columns = self.getInputCols()
        output_columns = self.getOutputCols()

        cols_to_process = [
            col for col in self.getInputCols() if str(dataset.schema[col].dataType) not in self._spark_numeric_types_str
        ]

        # TODO: Fix naming
        transformed_cols = [f"{self._fname_prefix}__{col}" for col in cols_to_process]

        model: LAMLStringIndexerModel = (
            self.indexer_model.setDefaultValue(self._fillna_val)
            .setHandleInvalid("keep")
            .setInputCols(cols_to_process)
            .setOutputCols(transformed_cols)
        )

        indexed_dataset = model.transform(dataset)

        for input_col, output_col in zip(input_columns, output_columns):
            if output_col in transformed_cols:
                continue
            indexed_dataset = indexed_dataset.withColumn(output_col, sf.col(input_col))

        indexed_dataset = indexed_dataset.replace(float("nan"), 0.0, subset=output_columns)

        logger.info(f"[{type(self)} (ORD)] Transform is finished")

        # output = self._make_output_df(indexed_dataset, self.getOutputCols())
        output = indexed_dataset

        return output


class SparkFreqEncoderEstimator(SparkLabelEncoderEstimator):
    """
    Calculates frequency in train data and
    produces :class:`~sparklightautoml.transformers.categorical.SparkFreqEncoderTransformer` instance.
    """

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "freq"

    _fillna_val = 1

    def __init__(self, input_cols: List[str], input_roles: RolesDict, do_replace_columns: bool = False):
        super().__init__(input_cols, input_roles, do_replace_columns, output_role=NumericRole(np.float32))

    def _fit(self, dataset: SparkDataFrame) -> "SparkFreqEncoderTransformer":
        logger.info(f"[{type(self)} (FE)] fit is started")

        indexer = LAMLStringIndexer(
            inputCols=self._input_intermediate_columns,
            outputCols=self.getOutputCols(),
            minFreqs=[1 for _ in self._input_intermediate_columns],
            handleInvalid="keep",
            defaultValue=self._fillna_val,
            freqLabel=True,  # Only for FREQ encoder
        )

        self.indexer_model = indexer.fit(dataset)

        logger.info(f"[{type(self)} (FE)] fit is finished")

        return SparkFreqEncoderTransformer(
            input_cols=self.getInputCols(),
            output_cols=self.getOutputCols(),
            input_roles=self.get_input_roles(),
            output_roles=self.get_output_roles(),
            do_replace_columns=self.get_do_replace_columns(),
            indexer_model=self.indexer_model,
        )


class SparkFreqEncoderTransformer(SparkLabelEncoderTransformer):
    """
    Labels are encoded with frequency in train data.

    Labels are integers from 1 to n.
    """

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "freq"
    _fillna_val = 1.0

    def __init__(
        self,
        input_cols: List[str],
        output_cols: List[str],
        input_roles: RolesDict,
        output_roles: RolesDict,
        do_replace_columns: bool,
        indexer_model: LAMLStringIndexerModel,
    ):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns, indexer_model)


class SparkCatIntersectionsHelper:
    """Helper class for :class:`~sparklightautoml.transformers.categorical.SparkCatIntersectionsEstimator` and
    :class:`~sparklightautoml.transformers.categorical.SparkCatIntersectionsTransformer`.
    """

    _fname_prefix = "inter"

    # noinspection PyMethodMayBeStatic
    def _make_col_name(self, cols: Sequence[str]) -> str:
        return f"({'__'.join(cols)})"

    def _make_category(self, cols: Sequence[str]) -> Column:
        lit = sf.lit("_")
        col_name = self._make_col_name(cols)
        columns_for_concat = []
        for col in cols:
            columns_for_concat.append(sf.col(col))
            columns_for_concat.append(lit)
        columns_for_concat = columns_for_concat[:-1]

        # return murmurhash3_32_udf(sf.concat(*columns_for_concat)).alias(col_name)
        return sf.hash(sf.concat(*columns_for_concat)).alias(col_name)

    def _build_df(
        self, df: SparkDataFrame, intersections: Optional[Sequence[Sequence[str]]]
    ) -> Tuple[SparkDataFrame, List[str]]:
        col_names = [self._make_col_name(comb) for comb in intersections]
        columns_to_select = [
            self._make_category(comb).alias(col_name) for comb, col_name in zip(intersections, col_names)
        ]
        df = df.select("*", *columns_to_select)
        return df, col_names


class SparkCatIntersectionsEstimator(SparkCatIntersectionsHelper, SparkLabelEncoderEstimator):
    """
    Combines categorical features
    and fits :class:`~sparklightautoml.transformers.categorical.SparkLabelEncoderEstimator`.
    Returns :class:`~sparklightautoml.transformers.categorical.SparkCatIntersectionsTransformer`.
    """

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "inter"

    def __init__(
        self,
        input_cols: List[str],
        input_roles: Dict[str, ColumnRole],
        intersections: Optional[Sequence[Sequence[str]]] = None,
        max_depth: int = 2,
        do_replace_columns: bool = False,
    ):
        super().__init__(
            input_cols,
            input_roles,
            do_replace_columns=do_replace_columns,
            output_role=CategoryRole(np.int32, label_encoded=True),
        )
        self.intersections = intersections
        self.max_depth = max_depth

        if self.intersections is None:
            self.intersections = []
            for i in range(2, min(self.max_depth, len(self.getInputCols())) + 1):
                self.intersections.extend(list(combinations(self.getInputCols(), i)))

        self._input_roles = {
            f"{self._make_col_name(comb)}": CategoryRole(
                np.int32,
                unknown=max((self.get_input_roles()[x].unknown for x in comb)),
                label_encoded=True,
            )
            for comb in self.intersections
        }
        self._input_columns = sorted(list(self._input_roles.keys()))

        out_roles = {f"{self._fname_prefix}__{f}": role for f, role in self._input_roles.items()}

        self.set(self.outputCols, list(out_roles.keys()))
        self.set(self.outputRoles, out_roles)

    def _fit(self, df: SparkDataFrame) -> Transformer:
        logger.info(f"[{type(self)} (CI)] fit is started")
        logger.debug(f"Calculating (CI) for input columns: {self.getInputCols()}")

        inter_df, inter_cols = self._build_df(df, self.intersections)

        self._input_intermediate_roles = {
            col: self.get_input_roles()[elts[0]] for col, elts in zip(inter_cols, self.intersections)
        }
        self._input_intermediate_columns = inter_cols

        super()._fit(inter_df)

        logger.info(f"[{type(self)} (CI)] fit is finished")

        return SparkCatIntersectionsTransformer(
            input_cols=self.getInputCols(),
            output_cols=self.getOutputCols(),
            input_roles=self.get_input_roles(),
            output_roles=self.get_output_roles(),
            do_replace_columns=self.get_do_replace_columns(),
            indexer_model=self.indexer_model,
            intersections=self.intersections,
        )


class SparkCatIntersectionsTransformer(SparkCatIntersectionsHelper, SparkLabelEncoderTransformer):
    """
    Combines category columns and encode with label encoder.
    """

    _fit_checks = (categorical_check,)
    _transform_checks = ()

    _fillna_val = 0

    def __init__(
        self,
        input_cols: List[str],
        output_cols: List[str],
        input_roles: RolesDict,
        output_roles: RolesDict,
        do_replace_columns: bool,
        indexer_model: LAMLStringIndexerModel,
        intersections: Optional[Sequence[Sequence[str]]] = None,
    ):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns, indexer_model)
        self.intersections = intersections

    def _transform(self, df: SparkDataFrame) -> SparkDataFrame:
        inter_df, self._input_intermediate_columns = self._build_df(df, self.intersections)

        out_df = super()._transform(inter_df)

        out_df = out_df.drop(*self._input_intermediate_columns)
        return out_df


class SparkOHEEncoderEstimator(SparkBaseEstimator):
    """
    Simple OneHotEncoder over label encoded categories.
    """

    _fit_checks = (categorical_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "ohe"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self.getInputCols()

    def __init__(
        self,
        input_cols: List[str],
        input_roles: Dict[str, ColumnRole],
        do_replace_columns: bool = False,
        make_sparse: Optional[bool] = None,
        total_feats_cnt: Optional[int] = None,
        dtype: type = np.float32,
    ):
        """
        Args:
            make_sparse: Create sparse matrix.
            total_feats_cnt: Initial features number.
            dtype: Dtype of new features.
        """
        super().__init__(input_cols, input_roles, do_replace_columns=do_replace_columns, output_role=None)

        self._make_sparse = make_sparse
        self._total_feats_cnt = total_feats_cnt
        self.dtype = dtype

        if self._make_sparse is None:
            assert self._total_feats_cnt is not None, "Param total_feats_cnt should be defined if make_sparse is None"

        self._ohe_transformer_and_roles: Optional[Tuple[Transformer, Dict[str, ColumnRole]]] = None

    def _fit(self, sdf: SparkDataFrame) -> Transformer:
        """Calc output shapes.
        Automatically do ohe in sparse form if approximate fill_rate < `0.2`.
        Args:
            sdf: Spark dataframe of categorical features.
        Returns:
            self.
        """

        maxs = [sf.max(c).alias(f"max_{c}") for c in self.getInputCols()]
        mins = [sf.min(c).alias(f"min_{c}") for c in self.getInputCols()]
        mm = sdf.select(maxs + mins).first().asDict()

        ohe = OneHotEncoder(inputCols=self.getInputCols(), outputCols=self.getOutputCols(), handleInvalid="error")
        transformer = ohe.fit(sdf)

        roles = {
            f"{self._fname_prefix}__{c}": NumericVectorOrArrayRole(
                size=mm[f"max_{c}"] - mm[f"min_{c}"] + 1,
                element_col_name_template=[
                    f"{self._fname_prefix}_{i}__{c}" for i in np.arange(mm[f"min_{c}"], mm[f"max_{c}"] + 1)
                ],
            )
            for c in self.getInputCols()
        }

        self._ohe_transformer_and_roles = (transformer, roles)

        return OHEEncoderTransformer(
            transformer,
            input_cols=self.getInputCols(),
            output_cols=self.getOutputCols(),
            input_roles=self.get_input_roles(),
            output_roles=roles,
        )


class OHEEncoderTransformer(SparkBaseTransformer, CommonPickleMLWritable, CommonPickleMLReadable):
    """OHEEncoder Transformer"""

    _fit_checks = (categorical_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "ohe"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self.getInputCols()

    def __init__(
        self,
        ohe_transformer: Transformer,
        input_cols: List[str],
        output_cols: List[str],
        input_roles: RolesDict,
        output_roles: RolesDict,
        do_replace_columns: bool = False,
    ):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)
        self._ohe_transformer = ohe_transformer

    def _transform(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """Transform categorical dataset to ohe.
        Args:
            sdf: Spark dataframe of categorical features.
        Returns:
            Numpy dataset with encoded labels.
        """
        output = self._ohe_transformer.transform(sdf)

        return output


def te_mapping_udf(broadcasted_dict):
    def f(folds, current_column):
        values_dict = broadcasted_dict.value
        try:
            return values_dict[f"{folds}_{current_column}"]
        except KeyError:
            return np.nan

    return sf.udf(f, "double")


class SparkTargetEncoderEstimator(SparkBaseEstimator):
    """
    Spark target encoder estimator.
    Returns :class:`~sparklightautoml.transformers.categorical.SparkTargetEncoderTransformer`.
    """

    _fit_checks = (categorical_check, oof_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "oof"

    def __init__(
        self,
        input_cols: List[str],
        input_roles: Dict[str, ColumnRole],
        alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 250.0, 1000.0),
        task_name: Optional[str] = None,
        folds_column: Optional[str] = None,
        target_column: Optional[str] = None,
        do_replace_columns: bool = False,
    ):
        super().__init__(
            input_cols, input_roles, do_replace_columns, NumericRole(np.float32, prob=task_name == "binary")
        )
        self.alphas = alphas
        self._task_name = task_name
        self._folds_column = folds_column
        self._target_column = target_column

    @staticmethod
    def score_func_binary(target, candidate) -> float:
        return -(target * np.log(candidate) + (1 - target) * np.log(1 - candidate))

    @staticmethod
    def score_func_reg(target, candidate) -> float:
        return (target - candidate) ** 2

    def _fit(self, dataset: SparkDataFrame) -> "SparkBaseTransformer":
        logger.info(f"[{type(self)} (TE)] fit_transform is started")

        assert self._target_column in dataset.columns, "Target should be presented in the dataframe"
        assert self._folds_column in dataset.columns, "Folds should be presented in the dataframe"

        self.encodings: Dict[str, np.ndarray] = dict()
        oof_feats_encoding: Dict[str, OOfFeatsMapping] = dict()

        sdf = dataset

        _fc = sf.col(self._folds_column)
        _tc = sf.col(self._target_column)

        logger.debug("Calculating totals (TE)")
        n_folds, prior, total_target_sum, total_count = sdf.select(
            sf.max(_fc) + 1, sf.mean(_tc.cast("double")), sf.sum(_tc).cast("double"), sf.count(_tc)
        ).first()

        logger.debug("Calculating folds priors (TE)")
        folds_prior_pdf = (
            sdf.groupBy(_fc)
            .agg(((total_target_sum - sf.sum(_tc)) / (total_count - sf.count(_tc))).alias("_folds_prior"))
            .collect()
        )

        def binary_score(col_name: str):
            return sf.mean(-(_tc * sf.log(col_name) + (1 - _tc) * sf.log(1 - sf.col(col_name)))).alias(col_name)

        def reg_score(col_name: str):
            return sf.mean(sf.pow((_tc - sf.col(col_name)), sf.lit(2))).alias(col_name)

        logger.debug("Starting processing features")
        feature_count = len(self.getInputCols())
        for i, feature in enumerate(self.getInputCols()):
            logger.debug(f"Processing feature {feature}({i}/{feature_count})")

            _cur_col = sf.col(feature)
            (dim_size,) = sdf.select((sf.max(_cur_col) + 1).astype("int").alias("dim_size")).first()

            logger.debug(f"Dim size of feature {feature}: {dim_size}")

            window_spec = Window.partitionBy(_cur_col)
            f_df = sdf.groupBy(_cur_col, _fc).agg(sf.sum(_tc).alias("f_sum"), sf.count(_tc).alias("f_count")).cache()

            oof_df = f_df.select(
                _cur_col,
                _fc,
                (sf.sum("f_sum").over(window_spec) - sf.col("f_sum")).alias("oof_sum"),
                (sf.sum("f_count").over(window_spec) - sf.col("f_count")).alias("oof_count"),
            )

            logger.debug(f"Creating maps column for fold priors (size={len(folds_prior_pdf)}) (TE)")
            mapping = {row[self._folds_column]: row["_folds_prior"] for row in folds_prior_pdf}
            folds_prior_exp = sf.create_map(*[sf.lit(x) for x in itertools.chain(*mapping.items())])

            logger.debug(f"Creating candidate columns (count={len(self.alphas)}) (TE)")
            candidates_cols = [
                ((sf.col("oof_sum") + sf.lit(alpha) * folds_prior_exp[_fc]) / (sf.col("oof_count") + sf.lit(alpha)))
                .cast("double")
                .alias(f"candidate_{i}")
                for i, alpha in enumerate(self.alphas)
            ]

            candidates_df = oof_df.select(_cur_col, _fc, *candidates_cols).cache()

            score_func = binary_score if self._task_name == "binary" else reg_score

            logger.debug("Calculating scores (TE)")
            scores = (
                sdf.join(candidates_df, on=[feature, self._folds_column])
                .select(*[score_func(f"candidate_{i}") for i, alpha in enumerate(self.alphas)])
                .first()
                .asDict()
            )
            logger.debug(f"Scores have been calculated (size={len(scores)}) (TE)")

            seq_scores = [scores[f"candidate_{i}"] for i, alpha in enumerate(self.alphas)]
            best_alpha_idx = np.argmin(seq_scores)
            best_alpha = self.alphas[best_alpha_idx]

            logger.debug("Collecting encodings (TE)")
            encoding_df = f_df.groupby(_cur_col).agg(
                ((sf.sum("f_sum") + best_alpha * prior) / (sf.sum("f_count") + best_alpha)).alias("encoding")
            )
            encoding = encoding_df.toPandas()
            logger.debug(f"Encodings have been collected (size={len(encoding)}) (TE)")
            f_df.unpersist()

            mapping = np.zeros(dim_size, dtype=np.float64)
            np.add.at(mapping, encoding[feature].astype(np.int32).to_numpy(), encoding["encoding"])
            self.encodings[feature] = mapping

            logger.debug("Collecting oof_feats (TE)")
            oof_feats_df = candidates_df.select(_cur_col, _fc, sf.col(f"candidate_{best_alpha_idx}").alias("encoding"))
            oof_feats = oof_feats_df.toPandas()
            logger.debug(f"oof_feats have been collected (size={len(oof_feats)}) (TE)")

            candidates_df.unpersist()

            mapping = np.zeros((n_folds, dim_size), dtype=np.float64)
            np.add.at(
                mapping,
                (
                    oof_feats[self._folds_column].astype(np.int32).to_numpy(),
                    oof_feats[feature].astype(np.int32).to_numpy(),
                ),
                oof_feats["encoding"],
            )

            oof_feats = OOfFeatsMapping(folds_column=self._folds_column, dim_size=dim_size, mapping=mapping)

            oof_feats_encoding[feature] = oof_feats

            logger.debug(f"[{type(self)} (TE)] Encodings have been calculated")

        logger.info(f"[{type(self)} (TE)] fit_transform is finished")

        return SparkTargetEncodeTransformer(
            tet=TargetEncoderTransformer.create(
                enc={col: mapping.tolist() for col, mapping in self.encodings.items()},
                oof_enc={col: mapping.mapping.tolist() for col, mapping in oof_feats_encoding.items()},
                fold_column=self._folds_column,
                apply_oof=True,
                input_cols=list(self.get_input_roles().keys()),
                output_cols=list(self.get_output_roles().keys()),
            ),
            input_roles=self.get_input_roles(),
            output_roles=self.get_output_roles(),
        )


def mcte_transform_udf(broadcasted_dict):
    def f(target, current_column):
        values_dict = broadcasted_dict.value
        try:
            return values_dict[(target, current_column)]
        except KeyError:
            return np.nan

    return sf.udf(f, "double")


class SparkMulticlassTargetEncoderEstimator(SparkBaseEstimator):
    """
    Spark multiclass target encoder estimator.
    Returns :class:`~sparklightautoml.transformers.categorical.SparkMultiTargetEncoderTransformer`.
    """

    _fit_checks = (categorical_check, multiclass_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "multioof"

    def __init__(
        self,
        input_cols: List[str],
        input_roles: Dict[str, ColumnRole],
        alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 250.0, 1000.0),
        task_name: Optional[str] = None,
        folds_column: Optional[str] = None,
        target_column: Optional[str] = None,
        do_replace_columns: bool = False,
    ):
        super().__init__(input_cols, input_roles, do_replace_columns, NumericRole(np.float32, prob=True))
        self.alphas = alphas
        self._task_name = task_name
        self._folds_column = folds_column
        self._target_column = target_column

    def _fit(self, dataset: SparkDataFrame) -> "SparkMultiTargetEncoderTransformer":
        logger.info(f"[{type(self)} (MCTE)] fit_transform is started")

        assert self._target_column in dataset.columns, "Target should be presented in the dataframe"
        assert self._folds_column in dataset.columns, "Folds should be presented in the dataframe"

        self.encodings = []

        df = dataset

        _fc = sf.col(self._folds_column)
        _tc = sf.col(self._target_column)

        tcn = self._target_column
        fcn = self._folds_column

        agg = df.groupBy([_fc, _tc]).count().toPandas().sort_values(by=[fcn, tcn])

        rows_count = agg["count"].sum()
        prior = agg.groupby(tcn).agg({"count": sum})

        prior["prior"] = prior["count"] / float(rows_count)
        prior = prior.to_dict()["prior"]

        agg["tt_sum"] = agg[tcn].map(agg[[tcn, "count"]].groupby(tcn).sum()["count"].to_dict()) - agg["count"]
        agg["tf_sum"] = rows_count - agg[fcn].map(agg[[fcn, "count"]].groupby(fcn).sum()["count"].to_dict())

        agg["folds_prior"] = agg["tt_sum"] / agg["tf_sum"]
        folds_prior_dict = agg[[fcn, tcn, "folds_prior"]].groupby([fcn, tcn]).max().to_dict()["folds_prior"]

        # Folds column unique values
        fcvs = sorted(list(set([fold for fold, target in folds_prior_dict.keys()])))
        # Target column unique values
        tcvs = sorted(list(set([target for fold, target in folds_prior_dict.keys()])))

        # cols_to_select = []

        for ccn in self.getInputCols():
            logger.debug(f"[{type(self)} (MCTE)] column {ccn}")

            _cc = sf.col(ccn)

            col_agg = df.groupby(_fc, _tc, _cc).count().toPandas()
            col_agg_dict = col_agg.groupby([ccn, fcn, tcn]).sum().to_dict()["count"]
            t_sum_dict = col_agg[[ccn, tcn, "count"]].groupby([ccn, tcn]).sum().to_dict()["count"]
            f_count_dict = col_agg[[ccn, fcn, "count"]].groupby([ccn, fcn]).sum().to_dict()["count"]
            t_count_dict = col_agg[[ccn, "count"]].groupby([ccn]).sum().to_dict()["count"]

            alphas_values = dict()
            # Current column unique values
            ccvs = sorted(col_agg[ccn].unique())

            for column_value in ccvs:
                for fold in fcvs:
                    oof_count = t_count_dict.get(column_value, 0) - f_count_dict.get((column_value, fold), 0)
                    for target in tcvs:
                        oof_sum = t_sum_dict.get((column_value, target), 0) - col_agg_dict.get(
                            (column_value, fold, target), 0
                        )
                        alphas_values[(column_value, fold, target)] = [
                            (oof_sum + a * folds_prior_dict[(fold, target)]) / (oof_count + a) for a in self.alphas
                        ]

            def make_candidates(x):
                fold, target, col_val, count = x
                values = alphas_values[(col_val, fold, target)]
                for i, a in enumerate(self.alphas):
                    x[f"alpha_{i}"] = values[i]
                return x

            candidates_df = col_agg.apply(make_candidates, axis=1)

            best_alpha_index = np.array(
                [
                    (-np.log(candidates_df[f"alpha_{i}"]) * candidates_df["count"]).sum()
                    for i, a in enumerate(self.alphas)
                ]
            ).argmin()

            column_encodings_dict = (
                pd.DataFrame(
                    [
                        [
                            ccv,
                            tcv,
                            (t_sum_dict.get((ccv, tcv), 0) + self.alphas[best_alpha_index] * prior[tcv])
                            / (t_count_dict[ccv] + self.alphas[best_alpha_index]),
                        ]
                        for (ccv, fcv, tcv), _ in alphas_values.items()
                    ],
                    columns=[ccn, tcn, "encoding"],
                )
                .groupby([tcn, ccn])
                .max()
                .to_dict()["encoding"]
            )

            self.encodings.append(column_encodings_dict)

        logger.info(f"[{type(self)} (MCTE)] fit_transform is finished")

        return SparkMultiTargetEncoderTransformer(
            encodings=self.encodings,
            input_cols=self.getInputCols(),
            input_roles=self.get_input_roles(),
            output_cols=self.getOutputCols(),
            output_roles=self.get_output_roles(),
            do_replace_columns=self.get_do_replace_columns(),
        )


class SparkMultiTargetEncoderTransformer(SparkBaseTransformer, CommonPickleMLWritable, CommonPickleMLReadable):
    """
    Spark multiclass target encoder transformer.
    """

    _fit_checks = (categorical_check, multiclass_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "multioof"

    def __init__(
        self,
        encodings: List[Dict[str, Any]],
        input_cols: List[str],
        output_cols: List[str],
        input_roles: RolesDict,
        output_roles: RolesDict,
        do_replace_columns: bool = False,
    ):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)
        self._encodings = encodings

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        cols_to_select = []
        logger.info(f"[{type(self)} (MCTE)] transform is started")

        sc = dataset.sql_ctx.sparkSession.sparkContext

        for i, (col_name, out_name) in enumerate(zip(self.getInputCols(), self.getOutputCols())):
            _cc = sf.col(col_name)
            logger.debug(f"[{type(self)} (MCTE)] transform map size for column {col_name}: {len(self._encodings[i])}")

            enc = self._encodings[i]
            values = sc.broadcast(enc)
            for tcv in {tcv for tcv, _ in enc.keys()}:
                cols_to_select.append(mcte_transform_udf(values)(sf.lit(tcv), _cc).alias(out_name))

        output = self._make_output_df(dataset, cols_to_select)

        logger.info(f"[{type(self)} (MCTE)] transform is finished")

        return output
