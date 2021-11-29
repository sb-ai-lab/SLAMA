from typing import Optional, Union, cast

from lightautoml.dataset.base import LAMLDataset
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures as LAMALinearFeatures
from lightautoml.pipelines.selection.base import ImportanceEstimator
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.features.base import TabularDataFeatures


# Same comments as for spark.pipelines.features.base
from lightautoml.spark.transformers.base import SequentialTransformer, SparkTransformer
from lightautoml.transformers.base import LAMLTransformer


class LinearFeatures(LAMALinearFeatures, TabularDataFeatures):
    """
    Creates pipeline for linear models and nnets.

    Includes:

        - Create categorical intersections.
        - OHE or embed idx encoding for categories.
        - Other cats to numbers ways if defined in role params.
        - Standartization and nan handling for numbers.
        - Numbers discretization if needed.
        - Dates handling.
        - Handling probs (output of lower level models).

    """

    def __init__(
        self,
        feats_imp: Optional[ImportanceEstimator] = None,
        top_intersections: int = 5,
        max_bin_count: int = 10,
        max_intersection_depth: int = 3,
        subsample: Optional[Union[int, float]] = None,
        sparse_ohe: Union[str, bool] = "auto",
        auto_unique_co: int = 50,
        output_categories: bool = True,
        multiclass_te_co: int = 3,
        **_
    ):
        """

        Args:
            feats_imp: Features importances mapping.
            top_intersections: Max number of categories
              to generate intersections.
            max_bin_count: Max number of bins to discretize numbers.
            max_intersection_depth: Max depth of cat intersection.
            subsample: Subsample to calc data statistics.
            sparse_ohe: Should we output sparse if ohe encoding
              was used during cat handling.
            auto_unique_co: Switch to target encoding if high cardinality.
            output_categories: Output encoded categories or embed idxs.
            multiclass_te_co: Cutoff if use target encoding in cat handling
              on multiclass task if number of classes is high.

        """
        super().__init__(
            feats_imp,
            top_intersections,
            max_bin_count,
            max_intersection_depth,
            subsample,
            sparse_ohe,
            auto_unique_co,
            output_categories,
            multiclass_te_co,
        )

    def _merge_seq(self, data: LAMLDataset) -> LAMLTransformer:
        data = cast(SparkDataset, data)
        pipes = []
        for pipe in self.pipes:
            _pipe = cast(SparkTransformer, pipe(data))

            with data.applying_temporary_caching():
                data = cast(SparkDataset, _pipe.fit_transform(data))
                data.cache_and_materialize()

            pipes.append(_pipe)

        return SequentialTransformer(pipes, is_already_fitted=True) if len(pipes) > 1 else pipes[-1]
