import pytest
import torch
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import TextRole, NumericRole, PathRole
from lightautoml.spark.transformers.text \
    import TfidfTextTransformer as SparkTfidfTextTransformer, AutoNLPWrap as SparkAutoNLPWrap

import pandas as pd
import numpy as np

from lightautoml.transformers.text import AutoNLPWrap
from .. import from_pandas_to_spark, spark, smoke_check, compare_by_content


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


def test_tokenizer(spark: SparkSession):
    source_data = pd.DataFrame(data={
        "a": ["This function is intended to compare", "two DataFrames and", "output any differences"],
        "b": ["Is is mostly intended ", "for use in unit tests", "Additional parameters allow "],
        "c": ["varying the strictness", "of the equality ", "checks performed"],
        "d": ["This example shows comparing", "two DataFrames that are equal", "but with columns of differing dtypes"]
    })

    ds = PandasDataset(source_data, roles={name: TextRole(np.str) for name in source_data.columns})

    from lightautoml.transformers.text import TokenizerTransformer
    from lightautoml.transformers.text import SimpleEnTokenizer

    lama_tokenizer_transformer = TokenizerTransformer(SimpleEnTokenizer(is_stemmer=False,to_string=False))
    lama_tokenizer_transformer.fit(ds)
    lama_result = lama_tokenizer_transformer.transform(ds)
    lama_result = lama_result.data
    print()
    print("lama_result")
    print(lama_result)

    from lightautoml.spark.transformers.text import Tokenizer as SparkTokenizer

    spark_tokenizer_transformer = SparkTokenizer()
    spark_dataset = from_pandas_to_spark(ds, spark)
    spark_tokenizer_transformer.fit(spark_dataset)
    spark_result = spark_tokenizer_transformer.transform(spark_dataset)
    spark_result = spark_result.to_pandas().data
    print("spark_result")
    print(spark_result)

    from pandas._testing import assert_frame_equal
    assert_frame_equal(lama_result, spark_result)

    # compare_by_content(spark, ds, lama_tokenizer_transformer, spark_tokenizer_transformer)


def test_concat_text_transformer(spark: SparkSession):
    source_data = pd.DataFrame(data={
        "a": ["This function is intended to compare", "two DataFrames and", "output any differences"],
        "b": ["Is is mostly intended ", "for use in unit tests", "Additional parameters allow "],
        "c": ["varying the strictness", "of the equality ", "checks performed"],
        "d": ["This example shows comparing", "two DataFrames that are equal", "but with columns of differing dtypes"]
    })

    ds = PandasDataset(source_data, roles={name: TextRole(np.str) for name in source_data.columns})

    from lightautoml.transformers.text import ConcatTextTransformer

    lama_transformer = ConcatTextTransformer()
    lama_transformer.fit(ds)
    lama_result = lama_transformer.transform(ds)
    lama_result = lama_result.data
    print()
    print("lama_result:")
    print(lama_result)

    from lightautoml.spark.transformers.text import ConcatTextTransformer as SparkConcatTextTransformer

    spark_tokenizer_transformer = SparkConcatTextTransformer()
    spark_dataset = from_pandas_to_spark(ds, spark)
    spark_tokenizer_transformer.fit(spark_dataset)
    spark_result = spark_tokenizer_transformer.transform(spark_dataset)
    spark_result = spark_result.to_pandas().data
    print()
    print("spark_result:")
    print(spark_result)

    from pandas._testing import assert_frame_equal
    assert_frame_equal(lama_result, spark_result)

    # compare_by_content(spark, ds, lama_tokenizer_transformer, spark_tokenizer_transformer)
