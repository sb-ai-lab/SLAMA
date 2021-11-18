import pytest
import torch
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import TextRole, NumericRole, PathRole
from lightautoml.spark.transformers.text \
    import TfidfTextTransformer as SparkTfidfTextTransformer, AutoNLPWrap as SparkAutoNLPWrap

import pandas as pd

from lightautoml.transformers.text import AutoNLPWrap
from . import smoke_check, compare_by_content, spark


@pytest.fixture
def text_dataset() -> PandasDataset:
    source_data = pd.DataFrame(data={
        "text_a": [f"Lorem Ipsum is simply dummy text of the printing and typesetting industry. {i}" for i in range(3)],
        "text_b": [f"Lorem Ipsum is simply dummy text of the printing and typesetting industry. {i}" for i in range(3)]
    })

    ds = PandasDataset(source_data, roles={name: PathRole() for name in source_data.columns})

    return ds


def test_tfidf_text_transformer(spark: SparkSession):
    param_defaults = {
        "min_df": 1.0,
        "max_df": 100.0,
        "max_features": 15
    }
    source_data = pd.DataFrame(data={
        "a": ["ipsen loren doloren" for _ in range(10)],
        "b": ["ipsen loren doloren" for _ in range(10)],
        "c": ["ipsen loren doloren" for _ in range(10)],
    })

    ds = PandasDataset(source_data, roles={name: TextRole() for name in source_data.columns})

    # we cannot compare by content because the formulas used by Spark and scikit is slightly different
    # see: https://github.com/scikit-learn/scikit-learn/blob/0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/feature_extraction/text.py#L1461
    # and: https://spark.apache.org/docs/latest/ml-features#tf-idf
    # we also cannot compare by metadata cause the number of resulting columns will be different
    # because of Spark and sklearn treats 'max_features' param for vocab size differently

    result_ds = smoke_check(spark, ds, SparkTfidfTextTransformer(param_defaults))

    assert len(result_ds.features) == len(source_data.columns) * param_defaults["max_features"]
    assert all(isinstance(r, NumericRole) for _, r in result_ds.roles.items())
    assert result_ds.shape[0] == source_data.shape[0]


@pytest.mark.skip
def test_auto_nlp_wrap(spark: SparkSession, text_dataset: PandasDataset):
    kwargs = {
        "model_name": "random_lstm",
        "device": torch.device("cpu:0"),
        "embedding_model": None
    }
    compare_by_content(spark, text_dataset,
                       AutoNLPWrap(**kwargs),
                       SparkAutoNLPWrap(**kwargs))
