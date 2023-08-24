import logging
import os

from typing import Dict
from typing import List
from uuid import uuid4

from lightautoml.dataset.base import RolesDict
from pyspark.ml.common import inherit_doc
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.param.shared import HasOutputCols
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.util import MLWriter
from pyspark.ml.wrapper import JavaParams
from pyspark.ml.wrapper import JavaTransformer

from sparklightautoml.mlwriters import CommonJavaToPythonMLReadable
from sparklightautoml.mlwriters import CommonPickleMLReadable
from sparklightautoml.mlwriters import CommonPickleMLWritable
from sparklightautoml.mlwriters import СommonPickleMLReader
from sparklightautoml.mlwriters import СommonPickleMLWriter
from sparklightautoml.transformers.base import SparkBaseTransformer
from sparklightautoml.utils import SparkDataFrame


logger = logging.getLogger(__name__)


@inherit_doc
class TargetEncoderTransformer(
    JavaTransformer, HasInputCols, HasOutputCols, CommonJavaToPythonMLReadable, JavaMLWritable
):
    """
    Scala-based implementation of Target Encoder transformer
    """

    @classmethod
    def create(
        cls,
        *,
        enc: Dict[str, List[float]],
        oof_enc: Dict[str, List[List[float]]],
        fold_column: str,
        apply_oof: bool,
        input_cols: List[str],
        output_cols: List[str],
    ):
        uid = f"TargetEncoderTransformer_{str(uuid4()).replace('-', '_')}"
        _java_obj = cls._new_java_obj(
            "org.apache.spark.ml.feature.lightautoml.TargetEncoderTransformer",
            uid,
            enc,
            oof_enc,
            fold_column,
            apply_oof,
        )

        tet = TargetEncoderTransformer(_java_obj).setInputCols(input_cols).setOutputCols(output_cols)
        return tet

    def __init__(self, java_obj):
        super(TargetEncoderTransformer, self).__init__()
        self._java_obj = java_obj

    def setInputCols(self, value) -> "TargetEncoderTransformer":
        self.set(self.inputCols, value)
        return self

    def setOutputCols(self, value) -> "TargetEncoderTransformer":
        self.set(self.outputCols, value)
        return self

    @staticmethod
    def _from_java(java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        Used for ML persistence.

        Meta-algorithms such as Pipeline should override this method as a classmethod.
        """

        def __get_class(clazz):
            """
            Loads Python class from its name.
            """
            parts = clazz.split(".")
            module = ".".join(parts[:-1])
            m = __import__(module)
            for comp in parts[1:]:
                m = getattr(m, comp)
            return m

        stage_name = "sparklightautoml.transformers.scala_wrappers.target_encoder_transformer.TargetEncoderTransformer"
        # Generate a default new instance from the stage_name class.
        py_type = __get_class(stage_name)
        if issubclass(py_type, JavaParams):
            # Load information from java_stage to the instance.
            py_stage = py_type(java_stage)
            # py_stage._java_obj = java_stage
            py_stage._resetUid(java_stage.uid())
            py_stage._transfer_params_from_java()
        elif hasattr(py_type, "_from_java"):
            py_stage = py_type._from_java(java_stage)
        else:
            raise NotImplementedError("This Java stage cannot be loaded into Python currently: %r" % stage_name)
        return py_stage


class SparkTargetEncoderTransformerMLWriter(СommonPickleMLWriter):
    """Implements saving an Estimator/Transformer instance to disk.
    Used when saving a trained pipeline.
    Implements MLWriter.saveImpl(path) method.
    """

    def __init__(self, instance: "SparkTargetEncodeTransformer"):
        super().__init__(instance)
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        super().saveImpl(path)
        tet_path = os.path.join(path, "scala_target_encoder_instance")
        self.instance._target_encoder_transformer.save(tet_path)


class SparkTargetEncoderTransformerMLReader(СommonPickleMLReader):
    def load(self, path) -> "SparkTargetEncodeTransformer":
        """Load the ML instance from the input path."""
        instance = super().load(path)
        tet_path = os.path.join(path, "scala_target_encoder_instance")
        instance._target_encoder_transformer = TargetEncoderTransformer.load(tet_path)

        return instance


class SparkTargetEncoderTransformerMLWritable(CommonPickleMLWritable):
    def write(self) -> MLWriter:
        assert isinstance(
            self, SparkTargetEncodeTransformer
        ), f"This class can work only with {type(SparkTargetEncodeTransformer)}"
        """Returns MLWriter instance that can save the Transformer instance."""
        return SparkTargetEncoderTransformerMLWriter(self)


class SparkTargetEncoderTransformerMLReadable(CommonPickleMLReadable):
    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return SparkTargetEncoderTransformerMLReader()


class SparkTargetEncodeTransformer(
    SparkBaseTransformer, SparkTargetEncoderTransformerMLWritable, SparkTargetEncoderTransformerMLReadable
):
    def __init__(self, tet: TargetEncoderTransformer, input_roles: RolesDict, output_roles: RolesDict):
        super(SparkTargetEncodeTransformer, self).__init__(
            list(input_roles.keys()), list(output_roles.keys()), input_roles, output_roles
        )

        self._target_encoder_transformer = tet

    def _transform(self, dataset: SparkDataFrame) -> SparkDataFrame:
        return self._target_encoder_transformer.transform(dataset)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["_target_encoder_transformer"]
        return state
