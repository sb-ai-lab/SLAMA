from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from typing import Iterator, Optional, Sequence, List, cast

import holidays
import numpy as np
import pandas as pd
from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import CategoryRole, NumericRole, ColumnRole, DatetimeRole
from lightautoml.transformers.datetime import datetime_check, date_attrs
from pyspark.ml.param.shared import Param, Params
from pyspark.sql import functions as sf, DataFrame as SparkDataFrame
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import IntegerType

from sparklightautoml.mlwriters import CommonPickleMLReadable, CommonPickleMLWritable
from sparklightautoml.transformers.base import SparkBaseTransformer


def get_unit_of_timestamp_column(seas: str, col: str):
    """Generates pyspark column to extract unit of time from timestamp

    Args:
        seas (str): unit of time: `y`(year), `m`(month), `d`(day),
        `wd`(weekday), `hour`(hour), `min`(minute), `sec`(second), `ms`(microsecond), `ns`(nanosecond)
        col (str): column name with datetime values
    """
    if seas == "y":
        return sf.year(sf.to_timestamp(sf.col(col)))
    elif seas == "m":
        return sf.month(sf.to_timestamp(sf.col(col)))
    elif seas == "d":
        return sf.dayofmonth(sf.to_timestamp(sf.col(col)))
    # TODO SPARK-LAMA: F.dayofweek() starts numbering from another day.
    # Differs from pandas.Timestamp.weekday.
    # elif seas == 'wd':
    #     return F.dayofweek(F.to_timestamp(F.col(col)))
    elif seas == "hour":
        return sf.hour(sf.to_timestamp(sf.col(col)))
    elif seas == "min":
        return sf.minute(sf.to_timestamp(sf.col(col)))
    elif seas == "sec":
        return sf.second(sf.to_timestamp(sf.col(col)))
    else:

        @pandas_udf(IntegerType())
        def get_timestamp_attr(arrs: Iterator[pd.Series]) -> Iterator[pd.Series]:
            for x in arrs:

                def convert_to_datetime(timestamp: int):
                    try:
                        date = pd.to_datetime(datetime.fromtimestamp(timestamp))
                    except:
                        date = datetime.now()
                    return date

                x = x.apply(lambda d: convert_to_datetime(d))
                yield getattr(x.dt, date_attrs[seas])

        return get_timestamp_attr(sf.to_timestamp(sf.col(col)).cast("long"))


class SparkDatetimeHelper:
    """
    Helper class for :class:`~sparklightautoml.transformers.datetime.SparkTimeToNumTransformer`,
    :class:`~sparklightautoml.transformers.datetime.SparkBaseDiffTransformer` and
    :class:`~sparklightautoml.transformers.datetime.SparkDateSeasonsTransformer`
    """

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
        "Y": 60 * 60 * 24 * 365,
    }

    _fit_checks = (datetime_check,)
    _transform_checks = ()


class SparkTimeToNumTransformer(
    SparkBaseTransformer, SparkDatetimeHelper, CommonPickleMLWritable, CommonPickleMLReadable
):
    """
    Transforms datetime columns values to numeric values.
    """

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
                (sf.to_timestamp(sf.col(i)).cast("long") - sf.to_timestamp(sf.lit(self.basic_time)).cast("long"))
                / self._interval_mapping[self.basic_interval]
            ).alias(out_col)
            new_cols.append(new_col)

        df = self._make_output_df(df, new_cols)

        return df


class SparkBaseDiffTransformer(
    SparkBaseTransformer, SparkDatetimeHelper, CommonPickleMLWritable, CommonPickleMLReadable
):
    """
    Basic conversion strategy, used in selection one-to-one transformers.
    Datetime converted to difference with basic_date.
    """

    _fname_prefix = "basediff"

    baseNames = Param(Params._dummy(), "baseNames", "base_names")

    diffNames = Param(Params._dummy(), "diffNames", "diff_names")

    basicInterval = Param(Params._dummy(), "basicInterval", "basic_interval")

    def __init__(
        self,
        input_roles: RolesDict,
        base_names: Sequence[str],
        diff_names: Sequence[str],
        basic_interval: Optional[str] = "D",
        do_replace_columns: bool = False,
    ):
        input_cols = list(base_names) + list(diff_names)

        self.base_names = base_names
        self.diff_names = diff_names
        self.basic_interval = basic_interval

        output_cols = [f"{self._fname_prefix}_{col}__{x}" for col in base_names for x in diff_names]

        output_roles = {col: NumericRole(dtype=np.float32) for col in output_cols}

        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)

        self.set(self.baseNames, self.base_names)
        self.set(self.diffNames, self.diff_names)
        self.set(self.basicInterval, self.basic_interval)

    def _transform(self, df: SparkDataFrame) -> SparkDataFrame:

        new_cols = [
            (
                (sf.to_timestamp(sf.col(dif)).cast("long") - sf.to_timestamp(sf.col(base)).cast("long"))
                / self._interval_mapping[self.basic_interval]
            ).alias(f"{self._fname_prefix}_{base}__{dif}")
            for base in self.base_names
            for dif in self.diff_names
        ]

        df = self._make_output_df(df, new_cols)

        return df


class SparkDateSeasonsTransformer(
    SparkBaseTransformer, SparkDatetimeHelper, CommonPickleMLWritable, CommonPickleMLReadable
):
    """
    Extracts unit of time from Datetime values and marks holiday dates.
    """

    _fname_prefix = "season"

    def __init__(
        self,
        input_cols: List[str],
        input_roles: RolesDict,
        do_replace_columns: bool = False,
        output_role: Optional[ColumnRole] = None,
    ):
        self.output_role = output_role
        if output_role is None:
            self.output_role = CategoryRole(np.int32)

        self.transformations = OrderedDict()
        output_cols = []
        for col in input_cols:
            rdt = cast(DatetimeRole, input_roles[col])
            seas = rdt.seasonality
            self.transformations[col] = seas
            for s in seas:
                output_cols.append(f"{self._fname_prefix}_{s}__{col}")
            if rdt.country is not None:
                output_cols.append(f"{self._fname_prefix}_hol__{col}")

        output_roles = {f: deepcopy(self.output_role) for f in output_cols}

        super().__init__(input_cols, output_cols, input_roles, output_roles, do_replace_columns)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        df = dataset
        roles = self.get_input_roles()

        new_cols = []
        for col in self.getInputCols():
            fcol = sf.to_timestamp(sf.col(col)).cast("long")
            seas_cols = [
                (
                    sf.when(sf.isnan(fcol) | sf.isnull(fcol), None)
                    .otherwise(get_unit_of_timestamp_column(seas, col))
                    .alias(f"{self._fname_prefix}_{seas}__{col}")
                )
                for seas in self.transformations[col]
            ]

            new_cols.extend(seas_cols)

            if roles[col].country is not None:

                @pandas_udf(IntegerType())
                def is_holiday(arrs: Iterator[pd.Series]) -> Iterator[pd.Series]:
                    for x in arrs:
                        x = x.apply(lambda d: datetime.fromtimestamp(d)).dt.normalize()
                        _holidays = holidays.country_holidays(
                            years=np.unique(x.dt.year.values),
                            country=roles[col].country,
                            prov=roles[col].prov,
                            state=roles[col].state,
                        )
                        yield x.isin(_holidays).astype(int)

                hol_col = is_holiday(fcol).alias(f"{self._fname_prefix}_hol__{col}")
                new_cols.append(hol_col)

        df = self._make_output_df(df, new_cols)

        return df
