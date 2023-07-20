"""Iterative feature selector."""

import logging

from typing import Iterator
from typing import Optional
from typing import cast

import numpy as np
import pandas as pd

from pandas import Series
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StructField

from sparklightautoml.pipelines.selection.base import SparkImportanceEstimator

from ...computations.base import ComputationsSettings
from ...computations.builder import build_computations_manager
from ...dataset.base import SparkDataset
from ...ml_algo.base import SparkTabularMLAlgo
from ...validation.base import SparkBaseTrainValidIterator


logger = logging.getLogger(__name__)


class SparkNpPermutationImportanceEstimator(SparkImportanceEstimator):
    """Permutation importance based estimator.

    Importance calculate, using random permutation
    of items in single column for each feature.

    """

    def __init__(self, random_state: int = 42, computations_settings: Optional[ComputationsSettings] = None):
        """
        Args:
            random_state: seed for random generation of features permutation.

        """
        super().__init__()
        self.random_state = random_state
        self._computations_manager = build_computations_manager(computations_settings)

    def fit(
        self,
        train_valid: Optional[SparkBaseTrainValidIterator] = None,
        ml_algo: Optional[SparkTabularMLAlgo] = None,
        preds: Optional[SparkDataset] = None,
    ):
        """Find importances for each feature in dataset.

        Args:
            train_valid: Initial dataset iterator.
            ml_algo: Algorithm.
            preds: Predicted target values for validation dataset.

        """
        logger.info(f"Starting importance estimating with {type(self)}")

        assert train_valid is not None, "train_valid cannot be None"

        normal_score = ml_algo.score(preds)
        logger.debug(f"Normal score = {normal_score}")

        valid_data = cast(SparkDataset, train_valid.get_validation_data())

        def build_score_func(it: int, feat: str):
            def func():
                logger.info(f"Start processing ({it},{feat})")
                df = valid_data.data

                field: StructField = df.schema[feat]

                @pandas_udf(returnType=field.dataType)
                def permutate(arrs: Iterator[pd.Series]) -> Iterator[pd.Series]:
                    permutator = np.random.RandomState(seed=self.random_state)
                    # one may get list of arrs and concatenate them to perform permutation
                    # in the whole partition
                    for x in arrs:
                        px = permutator.permutation(x)
                        yield pd.Series(px)

                permutated_df = df.withColumn(feat, permutate(feat))

                ds: SparkDataset = valid_data.empty()
                ds.set_data(permutated_df, valid_data.features, valid_data.roles, name=type(self).__name__)
                logger.debug("Dataframe with shuffled column prepared")

                # Calculate predict and metric
                new_preds = ml_algo.predict(ds)
                shuffled_score = ml_algo.score(new_preds)
                logger.debug(
                    "Shuffled score for col {} = {}, difference with normal = {}".format(
                        feat, shuffled_score, normal_score - shuffled_score
                    )
                )
                return feat, (normal_score - shuffled_score)

            return func

        results = self._computations_manager.compute(
            [build_score_func(it, feat) for it, feat in enumerate(valid_data.features)]
        )

        permutation_importance = {feat: diff_score for feat, diff_score in results}

        self.raw_importances = Series(permutation_importance).sort_values(ascending=False)

        logger.info(f"Finished importance estimating with {type(self)}")
