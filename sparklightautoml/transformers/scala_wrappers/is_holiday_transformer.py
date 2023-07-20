import logging

from typing import Dict
from typing import List
from typing import Set
from uuid import uuid4

from pyspark.ml.common import inherit_doc
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.param.shared import HasOutputCols
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaParams
from pyspark.ml.wrapper import JavaTransformer

from sparklightautoml.mlwriters import CommonJavaToPythonMLReadable


logger = logging.getLogger(__name__)


@inherit_doc
class IsHolidayTransformer(JavaTransformer, HasInputCols, HasOutputCols, CommonJavaToPythonMLReadable, JavaMLWritable):
    """
    Scala-based implementation of a transformer that checks whatever or not the date is a holiday
    """

    @classmethod
    def create(cls, *, holidays_dates: Dict[str, Set[str]], input_cols: List[str], output_cols: List[str]):
        uid = f"IsHolidayTransformer_{str(uuid4()).replace('-', '_')}"
        _java_obj = cls._new_java_obj(
            "org.apache.spark.ml.feature.lightautoml.IsHolidayTransformer", uid, holidays_dates
        )

        transformer = IsHolidayTransformer(_java_obj).setInputCols(input_cols).setOutputCols(output_cols)
        return transformer

    def __init__(self, java_obj):
        super(IsHolidayTransformer, self).__init__()
        self._java_obj = java_obj

    def setInputCols(self, value) -> "IsHolidayTransformer":
        self.set(self.inputCols, value)
        return self

    def setOutputCols(self, value) -> "IsHolidayTransformer":
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

        stage_name = "sparklightautoml.transformers.scala_wrappers.is_holiday_transformer.IsHolidayTransformer"
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
