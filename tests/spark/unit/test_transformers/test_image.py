import numpy as np
import pandas as pd
import pytest
import torch
from pyspark.sql import SparkSession

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import PathRole, NumericRole
from lightautoml.image.utils import pil_loader
from lightautoml.spark.transformers.image import ImageFeaturesTransformer as SparkImageFeaturesTransformer, \
    PathBasedAutoCVWrap as SparkPathBasedAutoCVWrap
from lightautoml.transformers.image import ImageFeaturesTransformer
from .. import spark, smoke_check, compare_by_content


@pytest.fixture
def image_dataset() -> PandasDataset:
    source_data = pd.DataFrame(data={
        "path_a": [f"unit/resources/images/cat_{i + 1}.jpg" for i in range(3)],
        "path_b": [f"unit/resources/images/cat_{i + 1}.jpg" for i in range(3)]
    })

    ds = PandasDataset(source_data, roles={name: PathRole() for name in source_data.columns})

    return ds


def test_path_auto_cv_wrap(spark: SparkSession, image_dataset: PandasDataset):
    result_ds = smoke_check(spark, image_dataset, SparkPathBasedAutoCVWrap(image_loader=pil_loader,
                                                                           device=torch.device("cpu:0")))

    # TODO: replace with a normal content check
    assert result_ds.shape[0] == image_dataset.shape[0]
    assert all(isinstance(role, NumericRole) and role.dtype == np.float32 for c, role in result_ds.roles.items())


def test_image_features_transformer(spark: SparkSession, image_dataset: PandasDataset):
    compare_by_content(spark, image_dataset,
                       ImageFeaturesTransformer(n_jobs=1, loader=pil_loader),
                       SparkImageFeaturesTransformer(n_jobs=1, loader=pil_loader))
