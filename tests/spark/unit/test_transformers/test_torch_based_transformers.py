import pytest
import torch
from pyspark.sql import SparkSession

import pandas as pd

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.roles import TextRole, PathRole
from lightautoml.image.utils import pil_loader
from lightautoml.spark.transformers.image import PathBasedAutoCVWrap as SparkPathBasedAutoCVWrap
from .test_transformers import smoke_check


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = SparkSession.builder.config("master", "local[1]").getOrCreate()

    print(f"Spark WebUI url: {spark.sparkContext.uiWebUrl}")

    yield spark

    spark.stop()


def test_auto_cv_wrap(spark: SparkSession):
    source_data = pd.DataFrame(data={
        "path_a": [f"resources/images/cat_{i + 1}.png" for i in range(3)],
        "path_b": [f"resources/images/cate_{i + 1}.png" for i in range(3)]
    })

    ds = PandasDataset(source_data, roles={name: PathRole() for name in source_data.columns})

    result_ds = smoke_check(spark, ds, SparkPathBasedAutoCVWrap(image_loader=pil_loader, device=torch.device("cpu:0")))



