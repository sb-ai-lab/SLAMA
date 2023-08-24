from typing import List
from typing import Optional
from typing import Union
from typing import cast

import numpy as np
import pandas as pd

from lightautoml.dataset.roles import CategoryRole
from lightautoml.reader.guess_roles import RolesDict
from lightautoml.reader.guess_roles import calc_ginis
from lightautoml.transformers.categorical import MultiClassTargetEncoder
from pyspark.ml import Pipeline
from pyspark.sql import functions as sf
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType

from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.transformers.base import SparkBaseTransformer
from sparklightautoml.transformers.base import SparkChangeRolesTransformer
from sparklightautoml.transformers.categorical import SparkFreqEncoderEstimator
from sparklightautoml.transformers.categorical import SparkLabelEncoderEstimator
from sparklightautoml.transformers.categorical import (
    SparkMulticlassTargetEncoderEstimator,
)
from sparklightautoml.transformers.categorical import SparkOrdinalEncoderEstimator
from sparklightautoml.transformers.categorical import SparkTargetEncoderEstimator
from sparklightautoml.transformers.numeric import SparkQuantileBinningEstimator


def get_gini_func(target_col: str):
    """Returns generator that take iterator by pandas dataframes and yield dataframes with calculated ginis.

    Args:
        target_col (str): target column to calc ginis
    """

    def gini_func(iterator):
        for pdf in iterator:
            target = pdf[target_col].to_numpy()
            data = pdf.drop(target_col, axis=1)
            cols = data.columns
            data = data.to_numpy()
            scores = calc_ginis(data, target, None)
            yield pd.DataFrame(data=[scores], columns=cols)

    return gini_func


def get_score_from_pipe(train: SparkDataset, pipe: Optional[Pipeline] = None) -> np.ndarray:
    """Get normalized gini index from pipeline.

    Args:
        train:  Spark dataset.
        pipe: Spark ML Estimator to obtain scores by means of.

    Returns:
        np.ndarray of scores.

    """

    if pipe is not None:
        pipeline_model = pipe.fit(train.data)
        df = pipeline_model.transform(train.data)
        last_stage = cast(SparkBaseTransformer, pipeline_model.stages[-1])

        sdf = df.select(SparkDataset.ID_COLUMN, train.target_column, *last_stage.getOutputCols())

        train = train.empty()
        train.set_data(sdf, features=last_stage.getOutputCols(), roles=last_stage.get_output_roles())

    gini_func = get_gini_func(train.target_column)

    # schema without target column
    output_schema = (train.data.select(*train.features)).schema
    # need to set True
    output_schema = StructType([StructField(f.name, f.dataType, True) for f in output_schema.fields])

    mean_scores = (
        (
            train.data.select(*train.features, train.target_column)
            .mapInPandas(gini_func, output_schema)
            .select([sf.mean(c).alias(c) for c in train.features])
        )
        .toPandas()
        .values.flatten()
    )

    return mean_scores


def get_numeric_roles_stat(
    train: SparkDataset,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
    manual_roles: Optional[RolesDict] = None,
) -> pd.DataFrame:
    """Calculate statistics about different encodings performances.

    We need it to calculate rules about advanced roles guessing.
    Only for numeric data.

    Args:
        train: Dataset.
        subsample: size of subsample.
        random_state: int.
        manual_roles: Dict.

    Returns:
        DataFrame.

    """
    if manual_roles is None:
        manual_roles = {}

    roles_to_identify = []
    roles = []
    flg_manual_set = []
    # check for train dtypes
    for f in train.features:
        role = train.roles[f]
        if role.name == "Numeric":
            roles_to_identify.append(f)
            roles.append(role)
            flg_manual_set.append(f in manual_roles)

    res = pd.DataFrame(
        columns=[
            "flg_manual",
            "unique",
            "unique_rate",
            "top_freq_values",
            "raw_scores",
            "binned_scores",
            "encoded_scores",
            "freq_scores",
            "nan_rate",
        ],
        index=roles_to_identify,
    )
    res["flg_manual"] = flg_manual_set

    if len(roles_to_identify) == 0:
        return res

    sdf = train.data.select(SparkDataset.ID_COLUMN, train.folds_column, train.target_column, *roles_to_identify)

    total_number = sdf.count()
    if subsample is not None:
        if subsample > total_number:
            fraction = 1.0
        else:
            fraction = subsample / total_number
        sdf = sdf.sample(fraction=fraction, seed=random_state)

    train = train.empty()
    train.set_data(sdf, roles_to_identify, roles)

    assert train.folds_column is not None

    # check task specific
    if train.task.name == "multiclass":
        encoder = SparkMulticlassTargetEncoderEstimator
    else:
        encoder = SparkTargetEncoderEstimator

    # check scores as is
    res["raw_scores"] = get_score_from_pipe(train)

    # check binned categorical score
    quantile_binning = SparkQuantileBinningEstimator(input_cols=train.features, input_roles=train.roles)
    target_encoder = encoder(
        input_cols=quantile_binning.getOutputCols(),
        input_roles=quantile_binning.get_output_roles(),
        task_name=train.task.name,
        folds_column=train.folds_column,
        target_column=train.target_column,
        do_replace_columns=True,
    )
    trf = Pipeline(stages=[quantile_binning, target_encoder])
    res["binned_scores"] = get_score_from_pipe(train, pipe=trf)

    # check label encoded scores
    change_roles = SparkChangeRolesTransformer(
        input_cols=train.features, input_roles=train.roles, role=CategoryRole(np.float32)
    )
    label_encoder = SparkLabelEncoderEstimator(
        input_cols=change_roles.getOutputCols(), input_roles=change_roles.get_output_roles(), random_state=random_state
    )
    target_encoder = encoder(
        input_cols=label_encoder.getOutputCols(),
        input_roles=label_encoder.get_output_roles(),
        task_name=train.task.name,
        folds_column=train.folds_column,
        target_column=train.target_column,
        do_replace_columns=True,
    )
    trf = Pipeline(stages=[change_roles, label_encoder, target_encoder])
    res["encoded_scores"] = get_score_from_pipe(train, pipe=trf)

    # check frequency encoding
    change_roles = SparkChangeRolesTransformer(
        input_cols=train.features, input_roles=train.roles, role=CategoryRole(np.float32)
    )
    freq_encoder = SparkFreqEncoderEstimator(
        input_cols=change_roles.getOutputCols(), input_roles=change_roles.get_output_roles()
    )
    trf = Pipeline(stages=[change_roles, freq_encoder])
    res["freq_scores"] = get_score_from_pipe(train, pipe=trf)

    nan_rate_cols = [sf.mean(sf.isnan(sf.col(feat)).astype(IntegerType())).alias(feat) for feat in train.features]
    res["nan_rate"] = train.data.select(nan_rate_cols).toPandas().values.flatten()

    return res


def get_category_roles_stat(
    train: SparkDataset, subsample: Optional[Union[float, int]] = 100000, random_state: int = 42
):
    """Search for optimal processing of categorical values.

    Categorical means defined by user or object types.

    Args:
        train: Dataset.
        subsample: size of subsample.
        random_state: seed of random numbers generator.

    Returns:
        result.

    """

    roles_to_identify = []

    dtypes = []

    # check for train dtypes
    roles = []
    for f in train.features:
        role = train.roles[f]
        if role.name == "Category" and role.encoding_type == "auto":
            roles_to_identify.append(f)
            roles.append(role)
            dtypes.append(role.dtype)

    res = pd.DataFrame(
        columns=[
            "unique",
            "top_freq_values",
            "dtype",
            "encoded_scores",
            "freq_scores",
            "ord_scores",
        ],
        index=roles_to_identify,
    )

    res["dtype"] = dtypes

    if len(roles_to_identify) == 0:
        return res

    sdf = train.data.select(SparkDataset.ID_COLUMN, train.folds_column, train.target_column, *roles_to_identify)

    if subsample is not None:
        total_number = sdf.count()
        if subsample > total_number:
            fraction = 1.0
        else:
            fraction = subsample / total_number
        sdf = sdf.sample(fraction=fraction, seed=random_state)

    train = train.empty()
    train.set_data(sdf, roles_to_identify, roles)

    assert train.folds_column is not None

    # check task specific
    if train.task.name == "multiclass":
        encoder = MultiClassTargetEncoder
    else:
        encoder = SparkTargetEncoderEstimator

    # check label encoded scores
    label_encoder = SparkLabelEncoderEstimator(
        input_cols=train.features, input_roles=train.roles, random_state=random_state
    )
    target_encoder = encoder(
        input_cols=label_encoder.getOutputCols(),
        input_roles=label_encoder.get_output_roles(),
        task_name=train.task.name,
        folds_column=train.folds_column,
        target_column=train.target_column,
        do_replace_columns=True,
    )
    trf = Pipeline(stages=[label_encoder, target_encoder])
    res["encoded_scores"] = get_score_from_pipe(train, pipe=trf)

    # check frequency encoding
    trf = SparkFreqEncoderEstimator(input_cols=train.features, input_roles=train.roles, do_replace_columns=True)
    trf = Pipeline(stages=[trf])
    res["freq_scores"] = get_score_from_pipe(train, pipe=trf)

    # check ordinal encoding
    trf = SparkOrdinalEncoderEstimator(input_cols=train.features, input_roles=train.roles, random_state=random_state)
    trf = Pipeline(stages=[trf])
    res["ord_scores"] = get_score_from_pipe(train, pipe=trf)

    return res


def get_null_scores(
    train: SparkDataset,
    feats: Optional[List[str]] = None,
    subsample: Optional[Union[float, int]] = 100000,
    random_state: int = 42,
) -> pd.Series:
    """Get null scores.

    Args:
        train: Dataset
        feats: list of features.
        subsample: size of subsample.
        random_state: seed of random numbers generator.

    Returns:
        Series.

    """
    roles = train.roles
    sdf = train.data.select(SparkDataset.ID_COLUMN, train.folds_column, train.target_column, *feats)

    if subsample is not None:
        total_number = sdf.count()
        if subsample > total_number:
            fraction = 1.0
        else:
            fraction = subsample / total_number
        sdf = sdf.sample(fraction=fraction, seed=random_state)

    train = train.empty()
    train.set_data(sdf, feats, [roles[f] for f in feats])

    train.data.cache()
    size = train.data.count()
    notnan = (
        train.data.select([sf.sum(sf.isnull(feat).astype(IntegerType())).alias(feat) for feat in train.features])
        .first()
        .asDict()
    )

    notnan_cols = [feat for feat, cnt in notnan.items() if cnt != size and cnt != 0]

    if notnan_cols:
        empty_slice_cols = [sf.when(sf.isnull(sf.col(feat)), 1.0).otherwise(0.0).alias(feat) for feat in notnan_cols]

        gini_func = get_gini_func(train.target_column)
        sdf = train.data.select(SparkDataset.ID_COLUMN, train.target_column, *empty_slice_cols)

        output_schema = sdf.select(SparkDataset.ID_COLUMN, *notnan_cols).schema

        mean_scores = (
            (sdf.mapInPandas(gini_func, output_schema).select([sf.mean(c).alias(c) for c in notnan_cols]))
            .first()
            .asDict()
        )
    else:
        mean_scores = {}

    scores = [mean_scores[feat] if feat in mean_scores else 0.0 for feat in train.features]

    train.data.unpersist()

    res = pd.Series(scores, index=train.features, name="max_score")

    return res
