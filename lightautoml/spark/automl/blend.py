from abc import ABC
from typing import Optional, Sequence, cast

import numpy as np
from pyspark.sql import functions as F

from lightautoml.automl.blend import Blender, \
    BestModelSelector as LAMABestModelSelector, \
    WeightedBlender as LAMAWeightedBlender
from lightautoml.dataset.roles import NumericRole
from lightautoml.spark.dataset.base import SparkDataset, SparkDataFrame
from lightautoml.spark.dataset.roles import NumericVectorOrArrayRole


class BlenderMixin(Blender, ABC):
    pass
    # def score(self, dataset: LAMLDataset) -> float:
    #     # TODO: SPARK-LAMA convert self._score to a required metric
    #
    #     raise NotImplementedError()


class BestModelSelector(BlenderMixin, LAMABestModelSelector):
    pass


class WeightedBlender(BlenderMixin, LAMAWeightedBlender):
    def _get_weighted_pred(self, splitted_preds: Sequence[SparkDataset], wts: Optional[np.ndarray]) -> SparkDataset:

        assert len(splitted_preds[0].features) == 1, \
            f"There should be only one feature containing predictions in the form of array, " \
            f"but: {splitted_preds[0].features}"

        feat = splitted_preds[0].features[0]
        role = splitted_preds[0].roles[feat]
        task = splitted_preds[0].task
        nan_feat = f"{feat}_nan_conf"
        # we put 0 here, because there cannot be more than output
        # even if works with vectors
        wfeat_name = "WeightedBlend_0"
        length = len(splitted_preds)
        if wts is None:
            # wts = np.ones(length, dtype=np.float32) / length
            wts = [1.0 / length for _ in range(length)]
        else:
            wts = [float(el) for el in wts]

        if task.name == "multiclass":
            assert isinstance(role, NumericVectorOrArrayRole), \
                f"The prediction should be an array or vector, but {type(role)}"

            vec_role = cast(NumericVectorOrArrayRole, role)
            wfeat_role = NumericVectorOrArrayRole(
                vec_role.size,
                f"WeightedBlend_{{}}",
                dtype=np.float32,
                prob=self._outp_prob,
                is_vector=vec_role.is_vector
            )

            def treat_nans(w: float):
                return [
                    F.transform(feat, lambda x: F.when(F.isnan(x), 0.0).otherwise(x * w)).alias(feat),
                    F.when(F.array_contains(feat, float('nan')), 0.0).otherwise(w).alias(nan_feat)
                ]

            def sum_predictions(summ_sdf: SparkDataFrame, curr_sdf: SparkDataFrame):
                return F.transform(F.arrays_zip(summ_sdf[feat], curr_sdf[feat]), lambda x, y: x + y).alias(feat)

            normalize_weighted_sum_col = (
                F.when(F.col(nan_feat) == 0.0, None)
                .otherwise(F.transform(feat, lambda x: x / F.col(nan_feat)))
                .alias(wfeat_name)
            )
        else:
            assert isinstance(role, NumericRole) and not isinstance(role, NumericVectorOrArrayRole), \
                f"The prediction should be numeric, but {type(role)}"

            wfeat_role = NumericRole(np.float32, prob=self._outp_prob)

            def treat_nans(w):
                return [
                    (F.col(feat) * w).alias(feat),
                    F.when(F.isnan(feat), 0.0).otherwise(w).alias(nan_feat)
                ]

            def sum_predictions(summ_sdf: SparkDataFrame, curr_sdf: SparkDataFrame):
                return (summ_sdf[feat] + curr_sdf[feat]).alias(feat)

            normalize_weighted_sum_col = (
                F.when(F.col(nan_feat) == 0.0, float('nan'))
                .otherwise(F.col(feat) / F.col(nan_feat))
                .alias(wfeat_name)
            )

        sum_with_nans_sdf = [
            x.data.select(
                SparkDataset.ID_COLUMN,
                *treat_nans(w)
            )
            for (x, w) in zip(splitted_preds, wts)
        ]

        sum_sdf = sum_with_nans_sdf[0]
        for sdf in sum_with_nans_sdf[1:]:
            sum_sdf = (
                sum_sdf
                .join(sdf, on=SparkDataset.ID_COLUMN)
                .select(
                    sum_sdf[SparkDataset.ID_COLUMN],
                    sum_predictions(sum_sdf, sdf),
                    (sum_sdf[nan_feat] + sdf[nan_feat]).alias(nan_feat)
                )
            )

        # TODO: SPARK-LAMA potentially this is a bad place check it later:
        #  1. equality condition double types
        #  2. None instead of nan (in the origin)
        #  due to Spark doesn't allow to mix types in the same column
        weighted_sdf = sum_sdf.select(
            SparkDataset.ID_COLUMN,
            normalize_weighted_sum_col
        )

        output = splitted_preds[0].empty()
        output.set_data(weighted_sdf, [wfeat_name], wfeat_role)

        return output
