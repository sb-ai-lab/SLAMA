from lightautoml.pipelines.ml.nested_ml_pipe import NestedTabularMLPipeline as LAMANestedTabularMLPipeline
from lightautoml.spark.pipelines.ml.base import SparkMLPipelineMixin


class NestedTabularMLPipeline(SparkMLPipelineMixin, LAMANestedTabularMLPipeline):
    """
        Same as NestedTabularMLPipeline of LAMA, but redefines a couple of methods via SparkMLPipelineMixin
    """
    pass
