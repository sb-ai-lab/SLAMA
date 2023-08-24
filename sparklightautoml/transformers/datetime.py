import itertools

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import cast

import holidays
import numpy as np
import pandas as pd

from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import CategoryRole
from lightautoml.dataset.roles import ColumnRole
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.dataset.roles import NumericRole
from lightautoml.transformers.datetime import date_attrs
from lightautoml.transformers.datetime import datetime_check
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param
from pyspark.ml.param.shared import Params
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as sf
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import IntegerType

from sparklightautoml.mlwriters import CommonPickleMLReadable
from sparklightautoml.mlwriters import CommonPickleMLWritable
from sparklightautoml.transformers.base import SparkBaseEstimator
from sparklightautoml.transformers.base import SparkBaseTransformer
from sparklightautoml.transformers.scala_wrappers.is_holiday_transformer import (
    IsHolidayTransformer,
)


_DateSeasonsTransformations = Dict[str, List[str]]


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


class SparkDateSeasonsEstimator(SparkBaseEstimator):
    _fname_prefix = "season"

    def __init__(self, input_cols: List[str], input_roles: RolesDict, output_role: Optional[ColumnRole] = None):
        output_role = output_role or CategoryRole(np.int32)

        super().__init__(input_cols, input_roles, output_role=output_role)
        self._infer_output_cols_and_roles(output_role)

    def _infer_output_cols_and_roles(self, output_role: ColumnRole):
        input_roles = self.get_input_roles()
        self.transformations: _DateSeasonsTransformations = OrderedDict()
        self.seasons_out_cols: Dict[str, List[str]] = dict()
        self.holidays_out_cols = []
        for col in self.getInputCols():
            rdt = cast(DatetimeRole, input_roles[col])
            seas = list(rdt.seasonality)
            self.transformations[col] = seas
            self.seasons_out_cols[col] = [f"{self._fname_prefix}_{s}__{col}" for s in seas]
            if rdt.country is not None:
                self.holidays_out_cols.append(f"{self._fname_prefix}_hol__{col}")

        output_cols = [*itertools.chain(*self.seasons_out_cols.values()), *self.holidays_out_cols]
        output_roles = {f: deepcopy(output_role) for f in output_cols}

        self.set(self.outputCols, output_cols)
        self.set(self.outputRoles, output_roles)

    def _fit(self, dataset: SparkDataFrame) -> Transformer:
        roles = self.get_input_roles()

        min_max_years = (
            dataset.select(
                *[
                    sf.struct(sf.year(sf.min(in_col)).alias("min"), sf.year(sf.max(in_col)).alias("max")).alias(in_col)
                    for in_col in self.getInputCols()
                ]
            )
            .first()
            .asDict()
        )

        holidays_cols_dates: Dict[str, Set[str]] = {
            col: set(
                dt.strftime("%Y-%m-%d")
                for dt in holidays.country_holidays(
                    years=list(range(min_y, max_y + 1)),
                    country=roles[col].country,
                    prov=roles[col].prov,
                    state=roles[col].state,
                ).keys()
            )
            for col, (min_y, max_y) in min_max_years.items()
        }

        return SparkDateSeasonsTransformer(
            input_cols=self.getInputCols(),
            seasons_out_cols=self.seasons_out_cols,
            holidays_out_cols=self.holidays_out_cols,
            input_roles=self.get_input_roles(),
            output_roles=self.get_output_roles(),
            seasons_transformations=self.transformations,
            holidays_dates=holidays_cols_dates,
        )


class SparkDateSeasonsTransformer(
    SparkBaseTransformer, SparkDatetimeHelper, CommonPickleMLWritable, CommonPickleMLReadable
):
    """
    Extracts unit of time from Datetime values and marks holiday dates.
    """

    _fname_prefix = "season"

    seasonOutCols = Param(Params._dummy(), "seasonOutCols", "seasonOutCols")
    holidaysOutCols = Param(Params._dummy(), "holidaysOutCols", "holidaysOutCols")
    seasonsTransformations = Param(Params._dummy(), "seasonsTransformations", "seasonsTransformations")
    holidaysDates = Param(Params._dummy(), "holidaysDates", "holidaysDates")

    def __init__(
        self,
        input_cols: List[str],
        seasons_out_cols: Dict[str, List[str]],
        holidays_out_cols: List[str],
        input_roles: RolesDict,
        output_roles: RolesDict,
        seasons_transformations: _DateSeasonsTransformations,
        holidays_dates: Dict[str, Set[str]],
    ):
        output_cols = [*itertools.chain(*seasons_out_cols.values()), *holidays_out_cols]
        super().__init__(input_cols, output_cols, input_roles, output_roles)

        self.set(self.seasonOutCols, seasons_out_cols)
        self.set(self.holidaysOutCols, holidays_out_cols)
        self.set(self.seasonsTransformations, seasons_transformations)
        self.set(self.holidaysDates, holidays_dates)

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        df = dataset

        seasons_out_cols = self.getOrDefault(self.seasonOutCols)
        holidays_out_cols = self.getOrDefault(self.holidaysOutCols)
        seasons_transformations = self.getOrDefault(self.seasonsTransformations)
        holidays_dates = self.getOrDefault(self.holidaysDates)

        new_cols = []
        for col, transformations in seasons_transformations.items():
            fcol = sf.to_timestamp(sf.col(col)).cast("long")
            seas_cols = [
                (
                    sf.when(sf.isnan(fcol) | sf.isnull(fcol), None)
                    .otherwise(get_unit_of_timestamp_column(seas, col))
                    .alias(out_col)
                )
                for out_col, seas in zip(seasons_out_cols[col], transformations)
            ]

            new_cols.extend(seas_cols)

        holidays_transformer = IsHolidayTransformer.create(
            holidays_dates=holidays_dates, input_cols=self.getInputCols(), output_cols=holidays_out_cols
        )
        df = holidays_transformer.transform(df.select("*", *new_cols))

        return df
