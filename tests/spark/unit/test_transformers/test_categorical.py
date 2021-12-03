from typing import cast, List

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset, NumpyDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.tasks.base import Task
from lightautoml.spark.transformers.categorical import LabelEncoder as SparkLabelEncoder, \
    FreqEncoder as SparkFreqEncoder, OrdinalEncoder as SparkOrdinalEncoder, \
    CatIntersectstions as SparkCatIntersectstions, OHEEncoder as SparkOHEEncoder, \
    TargetEncoder as SparkTargetEncoder, MultiClassTargetEncoder as SparkMultiClassTargetEncoder
from lightautoml.transformers.categorical import LabelEncoder, FreqEncoder, OrdinalEncoder, CatIntersectstions, \
    OHEEncoder
from lightautoml.tasks import Task

from . import compare_by_content, compare_by_metadata, DatasetForTest, spark, NumpyTransformable, compare_obtained_datasets
from lightautoml.spark.dataset.base import SparkDataset


DATASETS = [

    DatasetForTest("test_transformers/resources/datasets/dataset_23_cmc.csv", default_role=CategoryRole(np.int32)),

    DatasetForTest("test_transformers/resources/datasets/house_prices.csv",
                   columns=["Id", "MSSubClass", "MSZoning", "LotFrontage", "WoodDeckSF"],
                   roles={
                       "Id": CategoryRole(np.int32),
                       "MSSubClass": CategoryRole(np.int32),
                       "MSZoning": CategoryRole(str),
                       "LotFrontage": CategoryRole(np.float32),
                       "WoodDeckSF": CategoryRole(bool)
                   })
]


@pytest.mark.parametrize("dataset", DATASETS)
def test_label_encoder(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(spark, ds, LabelEncoder(), SparkLabelEncoder())


@pytest.mark.parametrize("dataset", DATASETS)
def test_freq_encoder(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(spark, ds, FreqEncoder(), SparkFreqEncoder())


@pytest.mark.parametrize("dataset", DATASETS)
def test_ordinal_encoder(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(spark, ds, OrdinalEncoder(), SparkOrdinalEncoder())


@pytest.mark.parametrize("dataset", DATASETS)
def test_cat_intersectstions(spark: SparkSession, dataset: DatasetForTest):

    ds = PandasDataset(dataset.dataset, roles=dataset.roles)

    compare_by_content(spark, ds, CatIntersectstions(), SparkCatIntersectstions())


def test_ohe(spark: SparkSession):
    make_sparse = False
    source_data = pd.DataFrame(data={
        "a": [1, 4, 5, 4, 2, 3],
        "b": [1, 4, 4, 4, 2, 3],
        "c": [1, 1, 1, 1, 1, 1],
        "d": [3, 1, 3, 2, 2, 1]
    })

    ds = PandasDataset(source_data, roles={
        name: CategoryRole(dtype=np.int32, label_encoded=True)
        for name in source_data.columns
    })
    _, _ = compare_by_metadata(spark, ds, OHEEncoder(make_sparse), SparkOHEEncoder(make_sparse))



@pytest.mark.parametrize("dataset", [DATASETS[1]])
def test_mock_target_encoder(spark: SparkSession, dataset: DatasetForTest):
    from lightautoml.transformers.categorical import TargetEncoder
    from lightautoml.spark.transformers.categorical import MockTargetEncoder

    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))

    label_encoder = LabelEncoder()
    label_encoder.fit(ds)
    labeled_ds = label_encoder.transform(ds)

    n_ds = NumpyDataset(
        distributed/master
        data=labeled_ds.data,
        features=labeled_ds.features,
        roles=labeled_ds.roles,
        task=labeled_ds.task,
        target=labeled_ds.data[:, -1],
        folds=labeled_ds.data[:, 2]
    )

# def test_target_encoder(spark: SparkSession):
#     df = pd.read_csv("test_transformers/resources/datasets/house_prices.csv")[
#         ["Id", 'MSSubClass', 'MSZoning', 'LotFrontage', 'WoodDeckSF']
#     ]
#     # %%
#     ds = PandasDataset(df.head(50),
#                        roles={
#                            "Id": CategoryRole(np.int32),
#                            "MSSubClass": CategoryRole(np.int32),
#                            "MSZoning": CategoryRole(str),
#                            "LotFrontage": CategoryRole(np.float32),
#                            "WoodDeckSF": CategoryRole(bool)
#                        },
#                        task=Task("binary")
#                        )

#     lt = LabelEncoder()
#     lt.fit(ds)
#     labeled_ds = lt.transform(ds)

#     ds = NumpyDataset(
#         data=labeled_ds.data,
#         features=labeled_ds.features,
#         roles=labeled_ds.roles,
#         task=labeled_ds.task,
#         target=labeled_ds.data[:, -1],
#         folds=labeled_ds.data[:, 2]
#     )

#     lama_transformer = TargetEncoder()
#     lama_result = lama_transformer.fit_transform(ds)

#     spark_data = from_pandas_to_spark(ds.to_pandas(), spark)
#     spark_data.task = Task("binary")
#     spark_transformer = SparkTargetEncoder()
#     spark_result = spark_transformer.fit_transform(spark_data, target_column='le__WoodDeckSF', folds_column='le__MSZoning')

#     lama_np_ds = lama_result.to_numpy()
#     spark_np_ds = spark_result.to_numpy()

#     assert list(sorted(lama_np_ds.features)) == list(sorted(spark_np_ds.features)), \
#         f"List of features are not equal\n" \
#         f"LAMA: {sorted(lama_np_ds.features)}\n" \
#         f"SPARK: {sorted(spark_np_ds.features)}"

#     # compare roles equality for the columns
#     assert lama_np_ds.roles == spark_np_ds.roles, "Roles are not equal"

#     # compare shapes
#     assert lama_np_ds.shape == spark_np_ds.shape, "Shapes are not equals"



# def test_multiclass_target_encoder(spark: SparkSession):
#     df = pd.read_csv("test_transformers/resources/datasets/house_prices.csv")[
#         ["Id", 'MSSubClass', 'MSZoning', 'LotFrontage', 'WoodDeckSF']
#     ]
#     # %%
#     ds = PandasDataset(df.head(50),
#                        roles={
#                            "Id": CategoryRole(np.int32),
#                            "MSSubClass": CategoryRole(np.int32),
#                            "MSZoning": CategoryRole(str),
#                            "LotFrontage": CategoryRole(np.float32),
#                            "WoodDeckSF": CategoryRole(np.int32)
#                        },
#                        task=Task("multiclass")
#                        )

#     lt = LabelEncoder()
#     lt.fit(ds)
#     labeled_ds = lt.transform(ds)

#     ds = NumpyDataset(

    sds = SparkDataset.from_lama(n_ds, spark)

    target_encoder = TargetEncoder()
    lama_output = target_encoder.fit_transform(n_ds)

    mock_encoder = MockTargetEncoder()
    mock_output = mock_encoder.fit_transform(sds)

    compare_obtained_datasets(lama_output, mock_output)

#     lama_transformer = MultiClassTargetEncoder()
#     lama_result = lama_transformer.fit_transform(ds)

#     spark_data = from_pandas_to_spark(ds.to_pandas(), spark)
#     spark_data.task = Task("multiclass")
#     spark_transformer = SparkMultiClassTargetEncoder()
#     spark_result = spark_transformer.fit_transform(spark_data, target_column='le__WoodDeckSF', folds_column='le__MSZoning')

#     lama_np_ds = lama_result.to_numpy()
#     spark_np_ds = spark_result.to_numpy()

#     assert list(sorted(lama_np_ds.features)) == list(sorted(spark_np_ds.features)), \
#         f"List of features are not equal\n" \
#         f"LAMA: {sorted(lama_np_ds.features)}\n" \
#         f"SPARK: {sorted(spark_np_ds.features)}"

#     # compare roles equality for the columns
#     assert lama_np_ds.roles == spark_np_ds.roles, "Roles are not equal"

#     # compare shapes
#     assert lama_np_ds.shape == spark_np_ds.shape, "Shapes are not equals"

