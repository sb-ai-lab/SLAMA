from copy import deepcopy
from typing import Dict, Optional, Sequence, List
from collections import defaultdict, OrderedDict
from itertools import chain, combinations
from datetime import datetime
import holidays
import numpy as np
import pandas as pd
from pandas import Series
from pyspark.sql import functions as F, types as SparkTypes, DataFrame as SparkDataFrame

from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import CategoryRole, NumericRole, ColumnRole
from lightautoml.transformers.datetime import datetime_check, date_attrs

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.transformers.base import ObsoleteSparkTransformer, SparkBaseTransformer

from pyspark.ml import Transformer, Estimator
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.param.shared import TypeConverters, Param, Params


def is_holiday(timestamp: int,
               country: str,
               state: Optional[str] = None,
               prov: Optional[str] = None) -> int:

    date = datetime.fromtimestamp(timestamp)
    return 1 if date in holidays.CountryHoliday(
        years=date.year,
        country=country,
        prov=prov,
        state=state
    ) else 0


def get_timestamp_attr(timestamp: int, attr: str) -> int:
    if not timestamp:
        return None

    try:
        date = pd.to_datetime(datetime.fromtimestamp(timestamp))
    except:
        date = datetime.now()

    try:
        at = getattr(date, attr)
        return at()
    except TypeError:
        return at


# TODO SPARK-LAMA: Replace with pandas_udf.
# They should be more efficient and arrow optimization is possible for them.
# https://github.com/fonhorst/LightAutoML/pull/57/files/57c15690d66fbd96f3ee838500de96c4637d59fe#r749551873
is_holiday_udf = F.udf(lambda *args, **kwargs: is_holiday(*args, **kwargs), SparkTypes.IntegerType())

# TODO SPARK-LAMA: It should to fail.
# https://github.com/fonhorst/LightAutoML/pull/57/files/57c15690d66fbd96f3ee838500de96c4637d59fe#r749610253
get_timestamp_attr_udf = F.udf(lambda *args, **kwargs: get_timestamp_attr(*args, **kwargs), SparkTypes.IntegerType())


class SparkDatetimeHelper:
    basic_interval = "D"

    _interval_mapping = {
        "NS": 0.000000001,
        "MS": 0.001,
        "SEC": 1,
        "MIN": 60,
        "HOUR": 60 * 60,
        "D": 60 * 60 * 24,

        # FIXME SPARK-LAMA: Very rough rounding
        "M": 60 * 60 * 24 * 30,
        "Y": 60 * 60 * 24 * 365
    }

    _fit_checks = (datetime_check,)
    _transform_checks = ()


class SparkTimeToNumTransformer(SparkBaseTransformer, SparkDatetimeHelper):
    basic_time = "2020-01-01"
    _fname_prefix = "dtdiff"

    def __init__(self, input_cols: List[str], input_roles: RolesDict, do_replace_columns: bool = False):
        output_cols = [f"{self._fname_prefix}__{f}" for f in input_cols]
        output_roles = {f: NumericRole(np.float32) for f in output_cols}
        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        df = dataset

        new_cols = []
        for i, out_col in zip(self.getInputCols(), self.getOutputCols()):
            new_col = (
                    (F.to_timestamp(F.col(i)).cast("long") - F.to_timestamp(F.lit(self.basic_time)).cast("long"))
                    / self._interval_mapping[self.basic_interval]
            ).alias(out_col)
            new_cols.append(new_col)

        df = self._make_output_df(df, new_cols)

        return df


class SparkBaseDiffTransformer(SparkBaseTransformer, SparkDatetimeHelper):
    _fname_prefix = "basediff"

    baseNames = Param(Params._dummy(), "baseNames",
                            "base_names")

    diffNames = Param(Params._dummy(), "diffNames",
                            "diff_names")

    basicInterval = Param(Params._dummy(), "basicInterval",
                            "basic_interval")

    def __init__(self,
                 input_roles: RolesDict,
                 base_names: Sequence[str],
                 diff_names: Sequence[str],
                 basic_interval: Optional[str] = "D",
                 do_replace_columns: bool = False):
        input_cols = list(base_names) + list(diff_names)

        self.base_names = base_names
        self.diff_names = diff_names
        self.basic_interval = basic_interval

        output_cols = [
            f"{self._fname_prefix}_{col}__{x}"
            for col in base_names
            for x in diff_names
        ]

        output_roles = {col: NumericRole(dtype=np.float32) for col in output_cols}

        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)

        self.set(self.baseNames, self.base_names)
        self.set(self.diffNames, self.diff_names)
        self.set(self.basicInterval, self.basic_interval)

    def _transform(self, df: SparkDataFrame) -> SparkDataFrame:

        new_cols = [
            ((
                    F.to_timestamp(F.col(dif)).cast("long") - F.to_timestamp(F.col(base)).cast("long")
            ) / self._interval_mapping[self.basic_interval]).alias(f"{self._fname_prefix}_{base}__{dif}")
            for base in self.base_names
            for dif in self.diff_names
        ]

        df = self._make_output_df(df, new_cols)

        return df


class SparkDateSeasonsTransformer(SparkBaseTransformer, SparkDatetimeHelper):
    _fname_prefix = "season"

    def __init__(self,
                 input_cols: List[str],
                 input_roles: RolesDict,
                 do_replace_columns: bool = False,
                 output_role: Optional[ColumnRole] = None):
        self.output_role = output_role
        if output_role is None:
            self.output_role = CategoryRole(np.int32)

        self.transformations = OrderedDict()
        output_cols = []
        for col in input_cols:
            seas = input_roles[col].seasonality
            self.transformations[col] = seas
            for s in seas:
                output_cols.append(f"{self._fname_prefix}_{s}__{col}")
            if input_roles[col].country is not None:
                output_cols.append(f"{self._fname_prefix}_hol__{col}")

        output_roles = {f: deepcopy(self.output_role) for f in output_cols}

        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        df = dataset
        roles = self.getInputRoles()

        new_cols = []
        for col in self.getInputCols():
            fcol = F.to_timestamp(F.col(col)).cast("long")
            seas_cols = [(
                F.when(F.isnan(fcol) | F.isnull(fcol), None)
                .otherwise(get_timestamp_attr_udf(fcol, F.lit(date_attrs[seas])))
                .alias(f"{self._fname_prefix}_{seas}__{col}")
            ) for seas in self.transformations[col]]

            new_cols.extend(seas_cols)

            if roles[col].country is not None:
                hol_col = (
                    is_holiday_udf(
                        fcol,
                        F.lit(roles[col].country),
                        F.lit(roles[col].state),
                        F.lit(roles[col].prov)
                    ).alias(f"{self._fname_prefix}_hol__{col}")
                )
                new_cols.append(hol_col)

        df = self._make_output_df(df, new_cols)

        return df
