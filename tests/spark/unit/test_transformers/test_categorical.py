import pickle

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset, NumpyDataset
from lightautoml.dataset.roles import CategoryRole
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.spark.reader.base import SparkToSparkReader
from lightautoml.spark.transformers.base import ColumnsSelector
from lightautoml.spark.transformers.categorical import LabelEncoder as SparkLabelEncoder, \
    FreqEncoder as SparkFreqEncoder, OrdinalEncoder as SparkOrdinalEncoder, \
    CatIntersectstions as SparkCatIntersectstions, OHEEncoder as SparkOHEEncoder, \
    TargetEncoder as SparkTargetEncoder
from lightautoml.spark.utils import print_exec_time
from lightautoml.tasks import Task
from lightautoml.transformers.categorical import LabelEncoder, FreqEncoder, OrdinalEncoder, CatIntersectstions, \
    OHEEncoder, TargetEncoder
from .. import DatasetForTest, from_pandas_to_spark, spark, compare_obtained_datasets, compare_by_metadata, \
    compare_by_content

DATASETS = [

    # DatasetForTest("unit/resources/datasets/dataset_23_cmc.csv", default_role=CategoryRole(np.int32)),

    DatasetForTest("unit/resources/datasets/house_prices.csv",
                   columns=["Id", "MSSubClass", "MSZoning", "LotFrontage", "WoodDeckSF"],
                   roles={
                       "Id": CategoryRole(np.int32),
                       "MSSubClass": CategoryRole(np.int32),
                       "MSZoning": CategoryRole(str),
                       "LotFrontage": CategoryRole(np.float32),
                       "WoodDeckSF": CategoryRole(bool)
                   })


    # DatasetForTest("unit/resources/datasets/house_prices.csv",
    #                columns=["Id", "MSZoning", "WoodDeckSF"],
    #                roles={
    #                    "Id": CategoryRole(np.int32),
    #                    "MSZoning": CategoryRole(str),
    #                    "WoodDeckSF": CategoryRole(bool)
    #                })
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

    # sds = SparkDataset.from_lama(ds, spark)
    sds = from_pandas_to_spark(ds, spark, ds.target)

    lama_transformer = CatIntersectstions()
    lama_transformer.fit(ds)
    lama_output = lama_transformer.transform(ds)

    spark_transformer = SparkCatIntersectstions()
    spark_transformer.fit(sds)
    spark_output = spark_transformer.transform(sds)

    compare_obtained_datasets(lama_output, spark_output)


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

    # ds = PandasDataset(dataset.dataset, roles=dataset.roles)
    _, _ = compare_by_metadata(spark, ds, OHEEncoder(make_sparse), SparkOHEEncoder(make_sparse))


@pytest.mark.parametrize("dataset", DATASETS)
def test_target_encoder(spark: SparkSession, dataset: DatasetForTest):
    ds = PandasDataset(dataset.dataset, roles=dataset.roles, task=Task("binary"))
# def test_target_encoder(spark: SparkSession):
#     with open("unit/resources/datasets/dataset_after_reader_dump.pickle", "rb") as f:
#         (data, features, roles, target) = pickle.load(f)
#
#     ds = PandasDataset(data, roles=roles, task=Task("binary"))
#     ds.target = target

    label_encoder = LabelEncoder()
    label_encoder.fit(ds)
    labeled_ds = label_encoder.transform(ds)

    cols = ["le__Id", "le__MSSubClass", "le__LotFrontage"]
    folds_col = "le__MSZoning"
    target_col = "le__WoodDeckSF"

    lpds = labeled_ds.to_pandas()
    _trg = lpds.data[target_col]
    _trg[_trg == 2] = 0

    n_ds = NumpyDataset(
        data=lpds.data[cols].to_numpy(),
        features=cols,
        roles=[labeled_ds.roles[col] for col in cols],
        task=labeled_ds.task,
        target=_trg,
        folds=lpds.data[folds_col].to_numpy()
    )
    n_ds = n_ds.to_pandas()
    # n_ds = labeled_ds.to_pandas()

    sds = from_pandas_to_spark(n_ds, spark, fill_folds_with_zeros_if_not_present=True)

    with print_exec_time():
        target_encoder = TargetEncoder()
        lama_output = target_encoder.fit_transform(n_ds)

    with print_exec_time():
        spark_encoder = SparkTargetEncoder()
        spark_output = spark_encoder.fit_transform(sds)

    compare_obtained_datasets(lama_output, spark_output)


# def test_target_encoder_2(spark: SparkSession):
#     df = spark.read.csv("../../examples/data/sampled_app_train.csv", header=True)
#
#     with print_exec_time():
#         sreader = SparkToSparkReader(task=Task("binary"), cv=5)
#         sds = sreader.fit_read(df)
#
#     feats_to_select = get_columns_by_role(sds, "Category")
#     with print_exec_time():
#         cs = ColumnsSelector(keys=feats_to_select)
#         cs_sds = cs.fit_transform(sds)
#
#     with print_exec_time():
#         slabel_encoder = SparkLabelEncoder()
#         labeled_sds = slabel_encoder.fit_transform(cs_sds)
#
#     with print_exec_time():
#         spark_encoder = SparkTargetEncoder()
#         spark_output = spark_encoder.fit_transform(labeled_sds)

# def test_multiclass_target_encoder(spark: SparkSession):
#     df = pd.read_csv("test_transformers/resources/datasets/house_prices.csv")[
#         ["Id", 'MSSubClass', 'MSZoning', 'LotFrontage', 'WoodDeckSF']
#     ]

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

    # ds = NumpyDataset(
    #     data=labeled_ds.data,
    #     features=labeled_ds.features,
    #     roles=labeled_ds.roles,
    #     task=labeled_ds.task,
    #     target=labeled_ds.data[:, -1],
    #     folds=labeled_ds.data[:, 2]
    # )

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

