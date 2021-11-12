from typing import List, Optional, Tuple, Dict

import numpy as np
from pyspark.ml import Transformer
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql import functions as F

from lightautoml.dataset.roles import ColumnRole
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole
from lightautoml.spark.transformers.base import SparkTransformer
from lightautoml.transformers.categorical import categorical_check, encoding_check


class OHEEncoder(SparkTransformer):
    """
    Simple OneHotEncoder over label encoded categories.
    """

    _fit_checks = (categorical_check, encoding_check)
    _transform_checks = ()
    _fname_prefix = "ohe"

    @property
    def features(self) -> List[str]:
        """Features list."""
        return self._features

    def __init__(
        self,
        make_sparse: Optional[bool] = None,
        total_feats_cnt: Optional[int] = None,
        dtype: type = np.float32,
    ):
        """

        Args:
            make_sparse: Create sparse matrix.
            total_feats_cnt: Initial features number.
            dtype: Dtype of new features.

        """
        self.make_sparse = make_sparse
        self.total_feats_cnt = total_feats_cnt
        self.dtype = dtype

        if self.make_sparse is None:
            assert self.total_feats_cnt is not None, "Param total_feats_cnt should be defined if make_sparse is None"

        self._ohe_transformer_and_roles: Optional[Tuple[Transformer, Dict[str, ColumnRole]]] = None

    def fit(self, dataset: SparkDataset):
        """Calc output shapes.

        Automatically do ohe in sparse form if approximate fill_rate < `0.2`.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            self.

        """

        # set transformer names and add checks
        for check_func in self._fit_checks:
            check_func(dataset)
        # set transformer features

        sdf = dataset.data
        temp_sdf = sdf.cache()
        maxs = [F.max(c).alias(f"max_{c}") for c in sdf.columns]
        mins = [F.min(c).alias(f"min_{c}") for c in sdf.columns]
        mm = temp_sdf.select(maxs + mins).collect()[0].asDict()

        self._features = [f"{self._fname_prefix}__{c}" for c in sdf.columns]

        ohe = OneHotEncoder(inputCols=sdf.columns, outputCols=self._features, handleInvalid="error")
        transformer = ohe.fit(temp_sdf)
        temp_sdf.unpersist()

        roles = {
            f"{self._fname_prefix}__{c}": NumericVectorOrArrayRole(
                size=mm[f"max_{c}"] - mm[f"min_{c}"] + 1,
                element_col_name_template=[
                    f"{self._fname_prefix}_{i}__{c}"
                    for i in np.arange(mm[f"min_{c}"], mm[f"max_{c}"] + 1)
                ]
            ) for c in sdf.columns
        }

        self._ohe_transformer_and_roles = (transformer, roles)

        return self

    def transform(self, dataset: SparkDataset) -> SparkDataset:
        """Transform categorical dataset to ohe.

        Args:
            dataset: Pandas or Numpy dataset of categorical features.

        Returns:
            Numpy dataset with encoded labels.

        """
        # checks here
        super().transform(dataset)

        sdf = dataset.data

        ohe, roles = self._ohe_transformer_and_roles

        # transform
        data = ohe.transform(sdf).select(list(roles.keys()))

        # create resulted
        output = dataset.empty()
        output.set_data(data, self.features, roles)

        return output
