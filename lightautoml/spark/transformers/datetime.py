from typing import Optional, Sequence, List
from collections import defaultdict, OrderedDict
from itertools import chain, combinations
from datetime import datetime
import holidays
import numpy as np
import pandas as pd
from pandas import Series
from pyspark.sql import functions as F, types as SparkTypes, DataFrame as SparkDataFrame

from lightautoml.dataset.roles import CategoryRole, NumericRole, ColumnRole
from lightautoml.transformers.datetime import datetime_check, date_attrs

from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.transformers.base import SparkTransformer


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

    date = pd.to_datetime(datetime.fromtimestamp(timestamp))
    at = getattr(date, attr)
    try:
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


class SparkDatetimeTransformer(SparkTransformer):

    basic_interval = "D"

    _interval_mapping = {
        "NS": 0.000000001,
        "MS": 0.001,
        "SEC": 1,
        "MIN": 60,
        "HOUR": 60*60,
        "D": 60*60*24,

        # FIXME SPARK-LAMA: Very rough rounding
        "M": 60*60*24*30,
        "Y": 60*60*24*365
    }

    _fit_checks = (datetime_check,)
    _transform_checks = ()


class TimeToNum(SparkDatetimeTransformer):

    basic_time = "2020-01-01"
    _fname_prefix = "dtdiff"

    def _transform(self, dataset: SparkDataset) -> SparkDataset:

        df = dataset.data

        # TODO SPARK-LAMA: It can be done easier without witColumn and withColumnRenamed.
        # Use .alias instead of the latter one.
        # https://github.com/fonhorst/LightAutoML/pull/57/files#r749549078
        for i in df.columns:
            df = df.withColumn(
                i,
                (
                    # TODO SPARK-LAMA: Spark gives wrong timestamp for parsed string even if we specify Timezone

                    # TODO SPARK-LAMA: Actually, you can just subtract a python var without 'F.to_timestamp(F.lit(self.basic_time)).cast("long")'
                    # basic_time_in_ts = datetime.to_unixtimestamp()
                    # (F.to_timestamp(i).cast("long") - basic_time_in_ts) / ...
                    # https://github.com/fonhorst/LightAutoML/pull/57/files#r749548681
                    F.to_timestamp(F.col(i)).cast("long") - F.to_timestamp(F.lit(self.basic_time)).cast("long")
                ) / self._interval_mapping[self.basic_interval]
            ).withColumnRenamed(i, f"{self._fname_prefix}__{i}")

        output = dataset.empty()
        output.set_data(df, self.features, NumericRole(np.float32))

        return output


class BaseDiff(SparkDatetimeTransformer):

    _fname_prefix = "basediff"

    @property
    def features(self) -> List[str]:
        return self._features

    def __init__(self,
                 base_names: Sequence[str],
                 diff_names: Sequence[str],
                 basic_interval: Optional[str] = "D"):

        self.base_names = base_names
        self.diff_names = diff_names
        self.basic_interval = basic_interval

    def _fit(self, dataset: SparkDataset) -> "SparkTransformer":

        # FIXME SPARK-LAMA: Возможно это можно будет убрать, т.к. у датасета будут колонки
        self._features = []
        for col in self.base_names:
            self._features.extend([f"{self._fname_prefix}_{col}__{x}" for x in self.diff_names])

        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:

        df = dataset.data

        for dif in self.diff_names:
            for base in self.base_names:
                df = df.withColumn(
                    f"{self._fname_prefix}_{base}__{dif}",
                    (
                        F.to_timestamp(F.col(dif)).cast("long") - F.to_timestamp(F.col(base)).cast("long")
                    ) / self._interval_mapping[self.basic_interval]
                )

        df = df.select(
            [f"{self._fname_prefix}_{base}__{dif}" for base in self.base_names for dif in self.diff_names]
        )

        output = dataset.empty()
        output.set_data(df, self.features, NumericRole(dtype=np.float32))

        return output


class DateSeasons(SparkDatetimeTransformer):

    _fname_prefix = "season"

    @property
    def features(self) -> List[str]:
        return self._features

    def __init__(self, output_role: Optional[ColumnRole] = None):

        self.output_role = output_role
        if output_role is None:
            self.output_role = CategoryRole(np.int32)

    def _fit(self, dataset: SparkDataset) -> "SparkTransformer":

        feats = dataset.features
        roles = dataset.roles
        self._features = []
        self.transformations = OrderedDict()

        for col in feats:
            seas = roles[col].seasonality
            self.transformations[col] = seas
            for s in seas:
                self._features.append(f"{self._fname_prefix}_{s}__{col}")
            if roles[col].country is not None:
                self._features.append(f"{self._fname_prefix}_hol__{col}")

        return self

    def _transform(self, dataset: SparkDataset) -> SparkDataset:

        df = dataset.data
        roles = dataset.roles

        for col in dataset.features:

            df = df.withColumn(col, F.to_timestamp(F.col(col)).cast("long"))

            for seas in self.transformations[col]:
                df = df \
                    .withColumn(
                        f"{self._fname_prefix}_{seas}__{col}",
                        F.when(F.isnan(F.col(col)), F.col(col))
                        .otherwise(
                            get_timestamp_attr_udf(F.col(col), F.lit(date_attrs[seas]))
                        )
                    )

            if roles[col].country is not None:
                df = df.withColumn(
                    f"{self._fname_prefix}_hol__{col}",
                    is_holiday_udf(
                        F.col(col),
                        F.lit(roles[col].country),
                        F.lit(roles[col].state),
                        F.lit(roles[col].prov)
                    )
                )

        df = df.select(
            [col for col in self.features]
        )

        output = dataset.empty()
        output.set_data(df, self.features, self.output_role)

        return output

