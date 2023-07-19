import functools
import logging.config
import uuid
from datetime import datetime
from typing import Union, Dict, cast, Tuple, List

import pandas as pd
from lightautoml.dataset.base import RolesDict
from lightautoml.dataset.roles import CategoryRole, DatetimeRole, NumericRole, ColumnRole
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as sf
from pyspark.sql.pandas.functions import pandas_udf

from examples_utils import get_spark_session
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_time

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


def generate_columns(col_enc: str, col_count: int) -> RolesDict:
    if col_enc == "freq":
        columns = {f"InColFreq_{i}": CategoryRole(object, encoding_type='freq') for i in range(col_count)}
    elif col_enc == "ord":
        columns = {f"InColOrd_{i}": CategoryRole(object, encoding_type='auto', ordinal=True) for i in range(col_count)}
    elif col_enc == "basediff":
        columns = {f"InColBaseDiff_{i}": DatetimeRole(base_date=True) for i in range(col_count)}
    elif col_enc == "LE":
        columns = {f"InColLE_{i}": CategoryRole(object, encoding_type='oof') for i in range(col_count)}
    elif col_enc == "ChRole":
        columns = {f"InColChRole_{i}": NumericRole(prob=True) for i in range(col_count)}
    elif col_enc == "LE#2":
        columns = {f"InColLE2_{i}": CategoryRole(object, encoding_type='auto') for i in range(col_count)}
    elif col_enc == "DateSeasons":
        columns = {f"InColDateSeasons_{i}": DatetimeRole(base_date=False) for i in range(col_count)}
    elif col_enc == "QB":
        columns = {f"InColQB_{i}": NumericRole(prob=False, discretization=True) for i in range(col_count)}
    elif col_enc == "regular":
        columns = {f"InColRegular_{i}": NumericRole(prob=False, discretization=False) for i in range(col_count)}
    else:
        raise Exception(f"Unknown col encoding: {col_enc}")

    return columns


def generate_placeholder_value(role: ColumnRole) -> Union[float, str, datetime]:
    if isinstance(role, NumericRole):
        return 42.0
    if isinstance(role, CategoryRole):
        return "category"
    if isinstance(role, DatetimeRole):
        return datetime.now()

    raise Exception(f"Unsupported type of ColumnRole: {type(role)}")


def generate_frame(
        cols: Union[Dict[str, int], int], rows_count: int,
        col_encs: List[str] = ('freq', 'ord', 'LE', 'ChRole', 'LE#2', 'DateSeasons', 'QB', 'regular')
) -> Tuple[SparkDataFrame, RolesDict]:
    if isinstance(cols, int):
        cols_mapping = {col_enc: cols for col_enc in col_encs}
    else:
        cols_mapping = cast(Dict[str, int], cols)

    all_cols_mapping = functools.reduce(
        lambda acc, x: {**acc, **x},
        (
            generate_columns(col_enc, col_count)
            for col_enc, col_count in cols_mapping.items()
        ),
        dict()
    )

    data = [
        {
            "_id": str(uuid.uuid4()),
            **{col_name: generate_placeholder_value(col_role) for col_name, col_role in all_cols_mapping.items()}
        }
        for _ in range(rows_count)
    ]

    data_sdf = spark.createDataFrame(data)

    h = 1.0 / 5
    folds_col = 'folds'
    target_col = 'target'
    data_sdf = data_sdf.select(
        "*",
        sf.floor(sf.rand(42) / h).alias(folds_col),
        sf.when(sf.rand(42) <= 0.5, sf.lit(0.0)).otherwise(sf.lit(1.0)).alias(target_col)
    )

    return data_sdf, all_cols_mapping


spark = get_spark_session()


@pandas_udf('string')
def test_add(s: pd.Series) -> pd.Series:
    return s.map(lambda x: f"{x}-{uuid.uuid4()}")


if __name__ == "__main__":

    ml_alg_kwargs = {
        'auto_unique_co': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': True,
        'top_intersections': 4
    }

    persistence_manager = PlainCachePersistenceManager()

    sdf, roles = generate_frame(cols=10, rows_count=100)

    with log_exec_time('initial_caching'):
        sdf = sdf.cache()
        sdf.write.mode('overwrite').format('noop').save()

    in_ds = SparkDataset(
        sdf,
        roles=roles,
        persistence_manager=persistence_manager,
        task=SparkTask("binary"),
        folds='folds',
        target='target',
        name="in_ds"
    )

    with log_exec_time():
        spark_features_pipeline = SparkLinearFeatures(**ml_alg_kwargs)
        out_ds = spark_features_pipeline.fit_transform(in_ds).persist()

    logger.info("Finished")

    out_ds.unpersist()

    spark.stop()
