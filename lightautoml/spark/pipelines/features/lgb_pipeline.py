from lightautoml.spark.pipelines.features.base import FeaturesPipeline
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline as LAMALGBAdvancedPipeline
from lightautoml.pipelines.utils import get_columns_by_role
from lightautoml.spark.dataset.base import SparkDataset
from lightautoml.spark.pipelines.features.base import TabularDataFeatures
from lightautoml.spark.transformers.base import SparkTransformer, SequentialTransformer, UnionTransformer, \
    ColumnsSelector
from lightautoml.spark.transformers.categorical import OrdinalEncoder
from lightautoml.spark.transformers.datetime import TimeToNum


class LGBSimpleFeatures(FeaturesPipeline):
    """Creates simple pipeline for tree based models.

    Simple but is ok for select features.
    Numeric stay as is, Datetime transforms to numeric.
    Categorical label encoding.
    Maps input to output features exactly one-to-one.

    """

    def create_pipeline(self, train: SparkDataset) -> SparkTransformer:
        """Create tree pipeline.

        Args:
            train: Dataset with train features.

        Returns:
            Composite datetime, categorical, numeric transformer.

        """
        # TODO: Transformer params to config
        transformers_list = []

        # process categories
        categories = get_columns_by_role(train, "Category")
        if len(categories) > 0:
            cat_processing = SequentialTransformer(
                [
                    ColumnsSelector(keys=categories),
                    OrdinalEncoder(subs=None, random_state=42),
                    # ChangeRoles(NumericRole(np.float32))
                ]
            )
            transformers_list.append(cat_processing)

        # process datetimes
        datetimes = get_columns_by_role(train, "Datetime")
        if len(datetimes) > 0:
            dt_processing = SequentialTransformer([ColumnsSelector(keys=datetimes), TimeToNum()])
            transformers_list.append(dt_processing)

        # process numbers
        numerics = get_columns_by_role(train, "Numeric")
        if len(numerics) > 0:
            num_processing = SequentialTransformer(
                [
                    ColumnsSelector(keys=numerics),
                    # ConvertDataset(dataset_type=NumpyDataset),
                ]
            )
            transformers_list.append(num_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all


class LGBAdvancedPipeline(TabularDataFeatures, LAMALGBAdvancedPipeline):
    pass
