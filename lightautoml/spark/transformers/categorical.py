import itertools
import logging
from collections import defaultdict
from itertools import combinations
from typing import Optional, Sequence, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from pandas import Series
from pyspark.ml import Transformer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.param.shared import Param, Params
from pyspark.sql import functions as F, types as SparkTypes, DataFrame as SparkDataFrame, Window, Column
from sklearn.utils.murmurhash import murmurhash3_32

from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import CategoryRole, NumericRole, ColumnRole
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.transformers.base import SparkBaseEstimator, SparkBaseTransformer
from lightautoml.transformers.categorical import categorical_check, encoding_check, oof_task_check, \
    multiclass_task_check

logger = logging.getLogger(__name__)


# FIXME SPARK-LAMA: np.nan in str representation is 'nan' while Spark's NaN is 'NaN'. It leads to different hashes.
# FIXME SPARK-LAMA: If udf is defined inside the class, it not works properly.
# "if murmurhash3_32 can be applied to a whole pandas Series, it would be better to make it via pandas_udf"
# https://github.com/fonhorst/LightAutoML/pull/57/files/57c15690d66fbd96f3ee838500de96c4637d59fe#r749534669
murmurhash3_32_udf = F.udf(lambda value: murmurhash3_32(value.replace("NaN", "nan"), seed=42) if value is not None else None, SparkTypes.IntegerType())


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
            "float128": "double"
        }
    )

    _spark_numeric_types = (
        SparkTypes.ShortType,
        SparkTypes.IntegerType,
        SparkTypes.LongType,
        SparkTypes.FloatType,
        SparkTypes.DoubleType,
        SparkTypes.DecimalType
    )


def pandas_dict_udf(broadcasted_dict):

    def f(s: Series) -> Series:
        values_dict = broadcasted_dict.value
        return s.map(values_dict)
    return F.pandas_udf(f, "double")


class SparkLabelEncoderEstimator(SparkBaseEstimator, TypesHelper):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "le"

    _fillna_val = 0

    def __init__(self,
                 input_cols: Optional[List[str]] = None,
                 input_roles: Optional[Dict[str, ColumnRole]] = None,
                 subs: Optional[float] = None,
                 random_state: Optional[int] = 42,
                 do_replace_columns: bool = False,
                 output_role: Optional[ColumnRole] = None):
        if not output_role:
            output_role = CategoryRole(np.int32, label_encoded=True)
        super().__init__(input_cols, input_roles, do_replace_columns=do_replace_columns, output_role=output_role)
        self._input_intermediate_columns = self.getInputCols()
        self._input_internediate_roles = self.getInputRoles()

    def _fit(self, dataset: SparkDataFrame) -> "SparkLabelEncoderTransformer":
        logger.info(f"[{type(self)} (LE)] fit is started")

        roles = self._input_internediate_roles

        df = dataset

        self.dicts = dict()

        for i in self._input_intermediate_columns:

            logger.debug(f"[{type(self)} (LE)] fit column {i}")

            role = roles[i]

            # TODO: think what to do with this warning
            co = role.unknown

            # FIXME SPARK-LAMA: Possible OOM point
            # TODO SPARK-LAMA: Can be implemented without multiple groupby and thus shuffling using custom UDAF.
            # May be an alternative it there would be performance problems.
            # https://github.com/fonhorst/LightAutoML/pull/57/files/57c15690d66fbd96f3ee838500de96c4637d59fe#r749539901
            vals = df \
                .groupBy(i).count() \
                .where(F.col("count") > co) \
                .select(i, F.col("count")) \
                .toPandas()

            logger.debug(f"[{type(self)} (LE)] toPandas is completed")

            vals = vals.sort_values(["count", i], ascending=[False, True])
            self.dicts[i] = Series(np.arange(vals.shape[0], dtype=np.int32) + 1, index=vals[i])
            logger.debug(f"[{type(self)} (LE)] pandas processing is completed")

        logger.info(f"[{type(self)} (LE)] fit is finished")

        return SparkLabelEncoderTransformer(input_cols=self.getInputCols(),
                                            output_cols=self.getOutputCols(),
                                            input_roles=self.getInputRoles(),
                                            output_roles=self.getOutputRoles(),
                                            do_replace_columns=self.getDoReplaceColumns(),
                                            dicts=self.dicts)


class SparkLabelEncoderTransformer(SparkBaseTransformer, TypesHelper):
    _transform_checks = ()
    _fname_prefix = "le"

    _fillna_val = 0

    fittedDicts = Param(Params._dummy(), "fittedDicts", "dicts from fitted Estimator")

    def __init__(self,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool,
                 dicts: Dict):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)
        self.set(self.fittedDicts, dicts)
        self._dicts = dicts
        self._input_columns = self.getInputCols()

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:

        logger.info(f"[{type(self)} (LE)] transform is started")

        df = dataset
        sc = df.sql_ctx.sparkSession.sparkContext

        cols_to_select = []

        for i, out_col in zip(self._input_columns,  self.getOutputCols()):
            logger.debug(f"[{type(self)} (LE)] transform col {i}")

            _ic = F.col(i)

            if i not in self._dicts:
                col = _ic
            elif len(self._dicts[i]) == 0:
                col = F.lit(self._fillna_val)
            else:
                vals: dict = self._dicts[i].to_dict()

                null_value = self._fillna_val
                if None in vals:
                    null_value = vals[None]
                    _ = vals.pop(None, None)

                if len(vals) == 0:
                    col = F.when(_ic.isNull(), null_value).otherwise(None)
                else:

                    nan_value = self._fillna_val

                    # if np.isnan(list(vals.keys())).any():  # not working
                    # Вот этот кусок кода тут по сути из-за OrdinalEncoder, который
                    # в КАЖДЫЙ dicts пихает nan. И вот из-за этого приходится его отсюда чистить.
                    # Нужно подумать, как это всё зарефакторить.
                    new_dict = {}
                    for key, value in vals.items():
                        try:
                            if np.isnan(key):
                                nan_value = value
                            else:
                                new_dict[key] = value
                        except TypeError:
                            new_dict[key] = value

                    vals = new_dict

                    if len(vals) == 0:
                        col = F.when(F.isnan(_ic), nan_value).otherwise(None)
                    else:
                        logger.debug(f"[{type(self)} (LE)] map size: {len(vals)}")

                        labels = sc.broadcast(vals)

                        if type(df.schema[i].dataType) in self._spark_numeric_types:
                            col = F.when(_ic.isNull(), null_value) \
                                .otherwise(
                                    F.when(F.isnan(_ic), nan_value)
                                     .otherwise(pandas_dict_udf(labels)(_ic))
                                )
                        else:
                            col = F.when(_ic.isNull(), null_value) \
                                   .otherwise(pandas_dict_udf(labels)(_ic))

            cols_to_select.append(col.alias(out_col))

        logger.info(f"[{type(self)} (LE)] Transform is finished")

        output = self._make_output_df(df, cols_to_select).fillna(self._fillna_val)

        return output


class SparkOrdinalEncoderEstimator(SparkLabelEncoderEstimator):
    _fit_checks = (categorical_check,)
    _fname_prefix = "ord"
    _fillna_val = float("nan")

    def __init__(self,
                 input_cols: Optional[List[str]] = None,
                 input_roles: Optional[Dict[str, ColumnRole]] = None,
                 subs: Optional[float] = None,
                 random_state: Optional[int] = 42):
        super().__init__(input_cols, input_roles, subs, random_state, output_role=NumericRole(np.float32))
        self.dicts = None
        self._use_cols = self.getInputCols()

    def _fit(self, dataset: SparkDataFrame) -> "Transformer":

        logger.info(f"[{type(self)} (ORD)] fit is started")

        # roles = dataset.roles
        roles = self.getOrDefault(self.inputRoles)

        self.dicts = {}
        for i in self._use_cols:

            logger.debug(f"[{type(self)} (ORD)] fit column {i}")

            role = roles[i]

            if not type(dataset.schema[i].dataType) in self._spark_numeric_types:

                co = role.unknown

                cnts = dataset \
                    .groupBy(i).count() \
                    .where((F.col("count") > co) & F.col(i).isNotNull()) \

                # TODO SPARK-LAMA: isnan raises an exception if column is boolean.
                if type(dataset.schema[i].dataType) != SparkTypes.BooleanType:
                    cnts = cnts \
                        .where(~F.isnan(F.col(i)))

                cnts = cnts \
                    .select(i) \
                    .toPandas()

                logger.debug(f"[{type(self)} (ORD)] toPandas is completed")

                cnts = Series(cnts[i].astype(str).rank().values, index=cnts[i])
                self.dicts[i] = cnts.append(Series([cnts.shape[0] + 1], index=[float("nan")])).drop_duplicates()
                logger.debug(f"[{type(self)} (ORD)] pandas processing is completed")

        logger.info(f"[{type(self)} (ORD)] fit is finished")

        return SparkOrdinalEncoderTransformer(input_cols=self.getInputCols(),
                                              output_cols=self.getOutputCols(),
                                              input_roles=self.getInputRoles(),
                                              output_roles=self.getOutputRoles(),
                                              do_replace_columns=self.getDoReplaceColumns(),
                                              dicts=self.dicts)


def ord_pandas_dict_udf(broadcasted_dict, fillna_value):

    def f(s: Series) -> Series:
        values_dict = broadcasted_dict.value
        fillna_val = fillna_value.value
        s[s == "nan"] = float("nan")
        s[s == None] = float("nan")
        return s.map(values_dict).fillna(fillna_val)
    return F.pandas_udf(f, "double")


class SparkOrdinalEncoderTransformer(SparkLabelEncoderTransformer):
    _transform_checks = ()
    _fname_prefix = "ord"
    _fillna_val = np.nan

    def __init__(self,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool,
                 dicts: Dict):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns, dicts)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:

        logger.info(f"[{type(self)} (ORD)] transform is started")

        df = dataset.fillna(self._fillna_val)
        sc = df.sql_ctx.sparkSession.sparkContext
        fill_na_val = sc.broadcast(self._fillna_val)

        cols_to_select = []

        for i, out_col in zip(self._input_columns,  self.getOutputCols()):
            logger.debug(f"[{type(self)} (ORD)] transform col {i}")

            _ic = F.col(i)

            if i not in self._dicts:
                col = _ic
            elif len(self._dicts[i]) == 0:
                col = F.lit(self._fillna_val)
            else:
                vals = self._dicts[i].to_dict()

                null_value = self._fillna_val
                if None in vals:
                    null_value = vals[None]
                    _ = vals.pop(None, None)

                if len(vals) == 0:
                    col = F.when(_ic.isNull(), null_value).otherwise(None)
                else:

                    nan_value = self._fillna_val

                    # if np.isnan(list(vals.keys())).any():  # not working
                    # Вот этот кусок кода тут по сути из-за OrdinalEncoder, который
                    # в КАЖДЫЙ dicts пихает nan. И вот из-за этого приходится его отсюда чистить.
                    # Нужно подумать, как это всё зарефакторить.
                    new_dict = {}
                    for key, value in vals.items():
                        try:
                            if np.isnan(key):
                                nan_value = value
                            else:
                                new_dict[key] = value
                        except TypeError:
                            new_dict[key] = value

                    vals = new_dict

                    if len(vals) == 0:
                        col = F.lit(nan_value)
                    else:
                        logger.debug(f"[{type(self)} (ORD)] map size: {len(vals)}")

                        labels = sc.broadcast(self._dicts[i].to_dict())

                        col = ord_pandas_dict_udf(labels, fill_na_val)(_ic).astype("double")

            cols_to_select.append(col.alias(out_col))

        logger.info(f"[{type(self)} (ORD)] Transform is finished")

        output = self._make_output_df(df, cols_to_select).fillna(self._fillna_val)

        return output


class SparkFreqEncoderEstimator(SparkLabelEncoderEstimator):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "freq"

    _fillna_val = 1

    def __init__(self,
                 input_cols: List[str],
                 input_roles: RolesDict,
                 do_replace_columns: bool = False):
        super().__init__(input_cols, input_roles, do_replace_columns, output_role=NumericRole(np.float32))

    def _fit(self, dataset: SparkDataFrame) -> "SparkFreqEncoderTransformer":

        logger.info(f"[{type(self)} (FE)] fit is started")

        df = dataset

        self.dicts = {}
        for i in self.getInputCols():

            logger.info(f"[{type(self)} (FE)] fit column {i}")

            vals = df \
                .groupBy(i).count() \
                .where(F.col("count") > 1) \
                .select(i, F.col("count")) \
                .toPandas()

            logger.debug(f"[{type(self)} (FE)] toPandas is completed")

            self.dicts[i] = vals.set_index(i)["count"]

            logger.debug(f"[{type(self)} (LE)] pandas processing is completed")

        logger.info(f"[{type(self)} (FE)] fit is finished")

        return SparkFreqEncoderTransformer(input_cols=self.getInputCols(),
                                           output_cols=self.getOutputCols(),
                                           input_roles=self.getInputRoles(),
                                           output_roles=self.getOutputRoles(),
                                           do_replace_columns=self.getDoReplaceColumns(),
                                           dicts=self.dicts)


class SparkFreqEncoderTransformer(SparkLabelEncoderTransformer):

    _fit_checks = (categorical_check,)
    _transform_checks = ()
    _fname_prefix = "freq"
    _fillna_val = 1

    def __init__(self,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool,
                 dicts: Dict):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns, dicts)


class SparkCatIntersectionsHelper:
    _fname_prefix = "inter"

    def _make_col_name(self, cols: Sequence[str]) -> str:
        return f"({'__'.join(cols)})"

    def _make_category(self, cols: Sequence[str]) -> Column:
        lit = F.lit("_")
        col_name = self._make_col_name(cols)
        columns_for_concat = []
        for col in cols:
            columns_for_concat.append(F.col(col))
            columns_for_concat.append(lit)
        columns_for_concat = columns_for_concat[:-1]

        return murmurhash3_32_udf(F.concat(*columns_for_concat)).alias(col_name)

    def _build_df(self, df: SparkDataFrame,
                  intersections: Optional[Sequence[Sequence[str]]]) -> SparkDataFrame:
        columns_to_select = [
            self._make_category(comb)
                .alias(f"{self._make_col_name(comb)}") for comb in intersections]
        df = df.select('*', *columns_to_select)
        return df


class SparkCatIntersectionsEstimator(SparkCatIntersectionsHelper, SparkLabelEncoderEstimator):

    _fit_checks = (categorical_check,)
    _transform_checks = ()

    def __init__(self,
                 input_cols: List[str],
                 input_roles: Dict[str, ColumnRole],
                 intersections: Optional[Sequence[Sequence[str]]] = None,
                 max_depth: int = 2,
                 do_replace_columns: bool = False):

        super().__init__(input_cols,
                         input_roles,
                         do_replace_columns=do_replace_columns,
                         output_role=CategoryRole(np.int32, label_encoded=True))
        self.intersections = intersections
        self.max_depth = max_depth

        if self.intersections is None:
            self.intersections = []
            for i in range(2, min(self.max_depth, len(self.getInputCols())) + 1):
                self.intersections.extend(list(combinations(self.getInputCols(), i)))

        self._input_roles = {
            f"{self._make_col_name(comb)}": CategoryRole(
                np.int32,
                unknown=max((self.getInputRoles()[x].unknown for x in comb)),
                label_encoded=True,
            ) for comb in self.intersections
        }
        self._input_columns = list(self._input_roles.keys())

        out_roles = {f"{self._fname_prefix}__{f}": role
                     for f, role in self._input_roles.items()}

        self.set(self.outputCols, list(out_roles.keys()))
        self.set(self.outputRoles, out_roles)

    def _fit(self, df: SparkDataFrame) -> Transformer:
        logger.info(f"[{type(self)} (CI)] fit is started")
        inter_df = self._build_df(df, self.intersections)

        super()._fit(inter_df)

        logger.info(f"[{type(self)} (CI)] fit is finished")

        return SparkCatIntersectionsTransformer(
            input_cols=self.getInputCols(),
            output_cols=self.getOutputCols(),
            input_roles=self.getInputRoles(),
            output_roles=self.getOutputRoles(),
            do_replace_columns=self.getDoReplaceColumns(),
            dicts=self.dicts,
            intersections=self.intersections
        )


class SparkCatIntersectionsTransformer(SparkCatIntersectionsHelper, SparkLabelEncoderTransformer):

    _fit_checks = (categorical_check,)
    _transform_checks = ()

    _fillna_val = 0

    def __init__(self,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool,
                 dicts: Dict,
                 intersections: Optional[Sequence[Sequence[str]]] = None):

        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns, dicts)
        self.intersections = intersections

    def _transform(self, df: SparkDataFrame) -> SparkDataFrame:
        inter_df = self._build_df(df, self.intersections)
        temp_cols = set(inter_df.columns).difference(df.columns)
        self._input_columns = temp_cols

        out_df = super()._transform(inter_df)

        out_df = out_df.drop(*temp_cols)
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
        return self._features

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
        super().__init__(input_cols,
                         input_roles,
                         do_replace_columns=do_replace_columns,
                         output_role=None)

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
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            self.

        """

        maxs = [F.max(c).alias(f"max_{c}") for c in self.getInputCols()]
        mins = [F.min(c).alias(f"min_{c}") for c in self.getInputCols()]
        mm = sdf.select(maxs + mins).first().asDict()

        ohe = OneHotEncoder(inputCols=self.getInputCols(), outputCols=self.getOutputCols(), handleInvalid="error")
        transformer = ohe.fit(sdf)

        roles = {
            f"{self._fname_prefix}__{c}": NumericVectorOrArrayRole(
                size=mm[f"max_{c}"] - mm[f"min_{c}"] + 1,
                element_col_name_template=[
                    f"{self._fname_prefix}_{i}__{c}"
                    for i in np.arange(mm[f"min_{c}"], mm[f"max_{c}"] + 1)
                ]
            ) for c in self.getInputCols()
        }

        self._ohe_transformer_and_roles = (transformer, roles)

        return OHEEncoderTransformer(
            transformer,
            input_cols=self.getInputCols(),
            output_cols=self.getOutputCols(),
            input_roles=self.getInputRoles(),
            output_roles=roles
        )


class OHEEncoderTransformer(SparkBaseTransformer):
    """OHEEncoder Transformer"""

    _fit_checks = (categorical_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "ohe"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(self,
                 ohe_transformer: Transformer,
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool = False):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)
        self._ohe_transformer = ohe_transformer

    def _transform(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """Transform categorical dataset to ohe.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

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
    return F.udf(f, "double")


class SparkTargetEncoderEstimator(SparkBaseEstimator):
    _fit_checks = (categorical_check, oof_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "oof"

    def __init__(self,
                 input_cols: List[str],
                 input_roles: Dict[str, ColumnRole],
                 alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 250.0, 1000.0),
                 task_name: Optional[str] = None,
                 folds_column: Optional[str] = None,
                 target_column: Optional[str] = None,
                 do_replace_columns: bool = False
                ):
        super().__init__(input_cols, input_roles, do_replace_columns,
                         NumericRole(np.float32, prob=task_name == "binary"))
        self.alphas = alphas
        self._task_name = task_name
        self._folds_column = folds_column
        self._target_column = target_column

    @staticmethod
    def score_func_binary(target, candidate) -> float:
        return -(
            target * np.log(candidate) + (1 - target) * np.log(1 - candidate)
        )

    @staticmethod
    def score_func_reg(target, candidate) -> float:
        return (target - candidate) ** 2

    def _fit(self, dataset: SparkDataFrame) -> "SparkTargetEncoderTransformer":
        logger.info(f"[{type(self)} (TE)] fit_transform is started")

        assert self._target_column in dataset.columns, "Target should be presented in the dataframe"
        assert self._folds_column in dataset.columns, "Folds should be presented in the dataframe"

        self.encodings = []

        sdf = dataset
        sc = sdf.sql_ctx.sparkSession.sparkContext

        _fc = F.col(self._folds_column)
        _tc = F.col(self._target_column)

        n_folds, prior, total_target_sum, total_count = sdf.select(
            F.max(_fc) + 1,
            F.mean(_tc.cast("double")),
            F.sum(_tc).cast("double"),
            F.count(_tc)
        ).first()

        folds_prior_pdf = sdf.groupBy(_fc).agg(
            ((total_target_sum - F.sum(_tc)) / (total_count - F.count(_tc))).alias("_folds_prior")
        ).collect()

        def binary_score(col_name: str):
            return F.mean(-(_tc * F.log(col_name) + (1 - _tc) * F.log(1 - F.col(col_name)))).alias(col_name)

        def reg_score(col_name: str):
            return F.mean(F.pow((_tc - F.col(col_name)), F.lit(2))).alias(col_name)

        for feature in self.getInputCols():
            _cur_col = F.col(feature)
            windowSpec = Window.partitionBy(_cur_col)
            f_df = sdf.groupBy(_cur_col, _fc).agg(F.sum(_tc).alias("f_sum"), F.count(_tc).alias("f_count")).cache()

            oof_df = (
                f_df
                    .select(
                    _cur_col,
                    _fc,
                    (F.sum('f_sum').over(windowSpec) - F.col("f_sum")).alias("oof_sum"),
                    (F.sum('f_count').over(windowSpec) - F.col("f_count")).alias("oof_count")
                )
            )

            mapping = {row[self._folds_column]: row["_folds_prior"] for row in folds_prior_pdf}
            folds_prior_exp = F.create_map(*[F.lit(x) for x in itertools.chain(*mapping.items())])

            candidates_cols = [
                ((F.col('oof_sum') + F.lit(alpha) * folds_prior_exp[_fc]) / (F.col('oof_count') + F.lit(alpha))).cast(
                    "double").alias(f"candidate_{i}")
                for i, alpha in enumerate(self.alphas)
            ]

            candidates_df = oof_df.select(_cur_col, _fc, *candidates_cols)

            score_func = binary_score if self._task_name == "binary" else reg_score

            scores = (
                sdf
                    .join(candidates_df, on=[feature, self._folds_column])
                    .select(*[score_func(f"candidate_{i}") for i, alpha in enumerate(self.alphas)])
                    .first()
                    .asDict()
            )

            seq_scores = [scores[f"candidate_{i}"] for i, alpha in enumerate(self.alphas)]
            best_alpha = self.alphas[np.argmin(seq_scores)]

            encoding = f_df.groupby(_cur_col).agg(
                ((F.sum("f_sum") + best_alpha * prior) / (F.sum('f_count') + best_alpha)).alias("encoding")).collect()
            f_df.unpersist()

            encoding = {row[feature]: row['encoding'] for row in encoding}

            self.encodings.append(encoding)

            logger.debug(f"[{type(self)} (TE)] Encodings have been calculated")

        logger.info(f"[{type(self)} (TE)] fit_transform is finished")

        return SparkTargetEncoderTransformer(
            encodings=self.encodings,
            input_cols=self.getInputCols(),
            input_roles=self.getInputRoles(),
            output_cols=self.getOutputCols(),
            output_roles=self.getOutputRoles(),
            do_replace_columns=self.getDoReplaceColumns()
        )


class SparkTargetEncoderTransformer(SparkBaseTransformer):

    _fit_checks = (categorical_check, oof_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "oof"

    def __init__(self,
                 encodings: List[Dict[str, Any]],
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool = False):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)
        self._encodings = encodings

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:

        cols_to_select = []
        logger.info(f"[{type(self)} (TE)] transform is started")

        sc = dataset.sql_ctx.sparkSession.sparkContext

        # TODO SPARK-LAMA: Нужно что-то придумать, чтобы ориентироваться по именам колонок, а не их индексу
        # Просто взять и забираться из dataset.features е вариант, т.к. в transform может прийти другой датасет
        # В оригинальной ламе об этом не парились, т.к. сразу переходили в numpy. Если прислали датасет не с тем
        # порядком строк - ну штоош, это проблемы того, кто датасет этот сюда вкинул. Стоит ли нам тоже придерживаться
        # этой логики?
        for i, (col_name, out_name) in enumerate(zip(self.getInputCols(), self.getOutputCols())):
            _cur_col = F.col(col_name)
            logger.debug(f"[{type(self)} (TE)] transform map size for column {col_name}: {len(self._encodings[i])}")

            values = sc.broadcast(self._encodings[i])

            cols_to_select.append(pandas_dict_udf(values)(_cur_col).alias(out_name))

        output = self._make_output_df(dataset, cols_to_select)

        logger.info(f"[{type(self)} (TE)] transform is finished")

        return output


# def mcte_mapping_udf(broadcasted_dict):
#     def f(folds, target, current_column):
#         values_dict = broadcasted_dict.value
#         try:
#             return values_dict[(folds, target, current_column)]
#         except KeyError:
#             return np.nan
#     return F.udf(f, "double")


def mcte_transform_udf(broadcasted_dict):
    def f(target, current_column):
        values_dict = broadcasted_dict.value
        try:
            return values_dict[(target, current_column)]
        except KeyError:
            return np.nan
    return F.udf(f, "double")


class SparkMulticlassTargetEncoderEstimator(SparkBaseEstimator):
    _fit_checks = (categorical_check, multiclass_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "multioof"

    def __init__(self,
                 input_cols: List[str],
                 input_roles: Dict[str, ColumnRole],
                 alphas: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 250.0, 1000.0),
                 task_name: Optional[str] = None,
                 folds_column: Optional[str] = None,
                 target_column: Optional[str] = None,
                 do_replace_columns: bool = False
                 ):
        super().__init__(input_cols, input_roles, do_replace_columns,
                         NumericRole(np.float32, prob=True))
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
        sc = df.sql_ctx.sparkSession.sparkContext

        _fc = F.col(self._folds_column)
        _tc = F.col(self._target_column)

        tcn = self._target_column
        fcn = self._folds_column

        agg = df.groupBy([_fc, _tc]).count().toPandas().sort_values(by=[fcn, tcn])

        rows_count = agg["count"].sum()
        prior = agg.groupby(tcn).agg({
            "count": sum
        })

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

            _cc = F.col(ccn)

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
                        oof_sum = t_sum_dict.get((column_value, target), 0) - col_agg_dict.get((column_value, fold, target), 0)
                        alphas_values[(column_value, fold, target)] = [(oof_sum + a * folds_prior_dict[(fold, target)]) / (oof_count + a) for a in self.alphas]

            def make_candidates(x):
                fold, target, column_value, count = x
                values = alphas_values[(column_value, fold, target)]
                for i, a in enumerate(self.alphas):
                    x[f"alpha_{i}"] = values[i]
                return x

            candidates_df = col_agg.apply(make_candidates, axis=1)

            best_alpha_index = np.array([(-np.log(candidates_df[f"alpha_{i}"]) * candidates_df["count"]).sum() for i, a in enumerate(self.alphas)]).argmin()

            # bacn = f"alpha_{best_alpha_index}"
            # processing_df = pd.DataFrame(
            #     [[fv, tv, cv, alp[best_alpha_index]] for (cv, fv, tv), alp in alphas_values.items()],
            #     columns=[fcn, tcn, ccn, bacn]
            # )

            # mapping = processing_df.groupby([fcn, tcn, ccn]).max().to_dict()[bacn]
            # values = sc.broadcast(mapping)

            # for tcv in tcvs:
            #     cols_to_select.append(mcte_mapping_udf(values)(_fc, F.lit(tcv), _cc).alias(f"{self._fname_prefix}_{tcv}__{ccn}"))

            column_encodings_dict = pd.DataFrame(
                [
                    [
                        ccv, tcv,
                        (t_sum_dict.get((ccv, tcv), 0) + self.alphas[best_alpha_index] * prior[tcv])
                        / (t_count_dict[ccv] + self.alphas[best_alpha_index])
                    ]
                    for (ccv, fcv, tcv), _ in alphas_values.items()
                ],
                columns=[ccn, tcn, "encoding"]
            ).groupby([tcn, ccn]).max().to_dict()["encoding"]

            self.encodings.append(column_encodings_dict)

        logger.info(f"[{type(self)} (MCTE)] fit_transform is finished")

        return SparkMultiTargetEncoderTransformer(
            encodings=self.encodings,
            input_cols=self.getInputCols(),
            input_roles=self.getInputRoles(),
            output_cols=self.getOutputCols(),
            output_roles=self.getOutputRoles(),
            do_replace_columns=self.getDoReplaceColumns()
        )


class SparkMultiTargetEncoderTransformer(SparkBaseTransformer):

    _fit_checks = (categorical_check, multiclass_task_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "multioof"

    def __init__(self,
                 encodings: List[Dict[str, Any]],
                 input_cols: List[str],
                 output_cols: List[str],
                 input_roles: RolesDict,
                 output_roles: RolesDict,
                 do_replace_columns: bool = False):
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)
        self._encodings = encodings

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:

        cols_to_select = []
        logger.info(f"[{type(self)} (MCTE)] transform is started")

        sc = dataset.sql_ctx.sparkSession.sparkContext

        for i, (col_name, out_name) in enumerate(zip(self.getInputCols(), self.getOutputCols())):
            _cc = F.col(col_name)
            logger.debug(f"[{type(self)} (MCTE)] transform map size for column {col_name}: {len(self._encodings[i])}")

            enc = self._encodings[i]
            values = sc.broadcast(enc)
            for tcv in {tcv for tcv, _ in enc.keys()}:
                cols_to_select.append(mcte_transform_udf(values)(F.lit(tcv), _cc).alias(out_name))

        output = self._make_output_df(dataset, cols_to_select)

        logger.info(f"[{type(self)} (MCTE)] transform is finished")

        return output
