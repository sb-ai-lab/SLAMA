from array import ArrayType
from copy import deepcopy
from typing import List, Optional, Dict, Tuple

from pyspark import Broadcast
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import FloatType

from lightautoml.image.image import DeepImageEmbedder
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorRole
from lightautoml.spark.transformers.base import SparkTransformer
from lightautoml.tasks.losses import torch
from lightautoml.text.utils import single_text_hash
from lightautoml.transformers.image import path_check

import pandas as pd
import numpy as np


class AutoCVWrap(SparkTransformer):
    """Calculate image embeddings."""

    _fit_checks = (path_check,)
    _transform_checks = ()
    _fname_prefix = "emb_cv"
    _emb_name = ""

    @property
    def features(self) -> List[str]:
        """Features list.

        Returns:
            List of features names.

        """
        return self._features

    def __init__(
        self,
        model="efficientnet-b0",
        weights_path: Optional[str] = None,
        cache_dir: str = "./cache_CV",
        subs: Optional = None,
        device: torch.device = torch.device("cuda:0"),
        n_jobs: int = 4,
        random_state: int = 42,
        is_advprop: bool = True,
        batch_size: int = 128,
        verbose: bool = True,
    ):
        """

        Args:
            model: Name of effnet model.
            weights_path: Path to saved weights.
            cache_dir: Path to cache directory or None.
            subs: Subsample to fit transformer. If ``None`` - full data.
            device: Torch device.
            n_jobs: Number of threads for dataloader.
            random_state: Random state to take subsample and set torch seed.
            is_advprop: Use adversarial training.
            batch_size: Batch size for embedding model.
            verbose: Verbose data processing.

        """
        self.embed_model = model
        self.random_state = random_state
        self.subs = subs
        self.cache_dir = cache_dir
        self._img_transformers: Optional[Dict[str, Tuple[DeepImageEmbedder, str]]] = None

        self.transformer = DeepImageEmbedder(
            device,
            n_jobs,
            random_state,
            is_advprop,
            model,
            weights_path,
            batch_size,
            verbose,
        )
        self._emb_name = "DI_" + single_text_hash(self.embed_model)
        self.emb_size = self.transformer.model.feature_shape

    def fit(self, dataset: SparkDataset):
        """Fit chosen transformer and create feature names.

        Args:
            dataset: Pandas or Numpy dataset of text features.

        """
        for check_func in self._fit_checks:
            check_func(dataset)

        # TODO: SPARK-LAMA cache should be implemented with an external storage system: Cassandra, Redis, S3, HDFS
        # if self.cache_dir is not None:
        #     if not os.path.exists(self.cache_dir):
        #         os.makedirs(self.cache_dir)
        # set transformer features

        sdf = dataset.data

        # TODO: SPARK-LAMA move subsampling into the base class method
        # fit
        # if self.subs is not None and df.shape[0] >= self.subs:
        #     subs = df.sample(n=self.subs, random_state=self.random_state)
        # else:
        #     subs = df

        self._img_transformers = dict()
        for c in sdf.columns:
            out_column_name = f"{self._fname_prefix}_{self._emb_name}__{c}"

            self._img_transformers[c] = (
                # TODO: we don't really want 'fit' here, because it would happen on the driver side
                # TODO: better to mark fitless classes with some Marker type via inheritance
                # TODO: to avoid errors of applying the wrong transformer as early as possible
                deepcopy(self.transformer.fit(sdf.select(c))),
                out_column_name
            )

        self._features = [feat for _, feat in self._img_transformers.values()]
        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform dataset to image embeddings.

        Args:
            dataset: Pandas or Numpy dataset of image paths.

        Returns:
            Numpy dataset with image embeddings.

        """
        # checks here
        super().transform(dataset)

        sdf = dataset.data

        # transform
        roles = NumericVectorRole(size=self.emb_size,
                                  element_col_name_template=f"{self._fname_prefix}_{self._emb_name}_{{}}__{c}",
                                  dtype=np.float32)
        outputs = []

        new_cols = []
        for c in sdf.columns:
            trans, out_col_name = self._img_transformers[c]
            transformer_bcast = Broadcast(value=trans)

            @pandas_udf(ArrayType(ArrayType(FloatType())))
            def calculate_embeddings(paths: pd.Series):
                transformer = transformer_bcast.value
                img_embeds = pd.Series([transformer.transform(p) for p in paths])
                return img_embeds

            new_cols.append(calculate_embeddings(c).alias(out_col_name))

        new_sdf = sdf.select(new_cols)

        output = dataset.empty()
        output.set_data(new_sdf, self.features, roles)

        return output