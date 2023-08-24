"""
The following methods have been copied from the HandySpark project and modified.
https://github.com/dvgodoy/handyspark
"""
from pyspark.mllib.common import _java2py
from pyspark.mllib.common import _py2java


# noinspection PyProtectedMember
def call2(obj, name, *a):
    serde = obj._sc._jvm.org.apache.spark.mllib.api.python.SerDe
    args = [_py2java(obj._sc, a) for a in a]
    java_res = getattr(obj._java_model, name)(*args)
    java_res = serde.fromTuple2RDD(java_res)
    res = _java2py(obj._sc, java_res)
    return res
