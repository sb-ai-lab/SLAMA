import pandas as pd
import pytest
import torch
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import PathRole
from lightautoml.image.utils import pil_loader
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.transformers.image import PathBasedAutoCVWrap as SparkPathBasedAutoCVWrap, \
    ImageFeaturesTransformer as SparkImageFeaturesTransformer
from lightautoml.transformers.image import ImageFeaturesTransformer
from . import compare_by_content
from .test_transformers import smoke_check


@pytest.fixture
def image_dataset() -> PandasDataset:
    source_data = pd.DataFrame(data={
        "path_a": [f"resources/images/cat_{i + 1}.jpg" for i in range(3)],
        "path_b": [f"resources/images/cat_{i + 1}.jpg" for i in range(3)]
    })

    ds = PandasDataset(source_data, roles={name: PathRole() for name in source_data.columns})

    return ds

@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = SparkSession.builder.config("master", "local[1]").getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


@pytest.mark.skip()
def test_path_auto_cv_wrap(spark: SparkSession, image_dataset: PandasDataset):
    result_ds = smoke_check(spark, image_dataset, SparkPathBasedAutoCVWrap(image_loader=pil_loader, device=torch.device("cpu:0")))

    assert result_ds.shape == image_dataset.shape
    assert all(isinstance(role, NumericVectorOrArrayRole) for c, role in result_ds.roles.items())
    # TODO: add content check


def test_image_features_transformer(spark: SparkSession, image_dataset: PandasDataset):
    compare_by_content(spark, image_dataset,
                       ImageFeaturesTransformer(n_jobs=1, loader=pil_loader),
                       SparkImageFeaturesTransformer(n_jobs=1, loader=pil_loader))



# def test_array_auto_cv_wrap(spark: SparkSession):
#     source_data = pd.DataFrame(data={
#         "path_a": [np.array(pil_loader(f"resources/images/cat_{i + 1}.jpg")) for i in range(3)],
#         "path_b": [np.array(pil_loader(f"resources/images/cat_{i + 1}.jpg")) for i in range(3)]
#     })
#
#     ds = PandasDataset(source_data, roles={name: NumericVectorOrArrayRole(None, None, is_vector=False) for name in source_data.columns})
#
#     result_ds = smoke_check(spark, ds, SparkArrayBasedAutoCVWrap(device=torch.device("cpu:0")))


