from importlib_metadata import version
from packaging.version import parse
from pyspark import SparkContext
from pyspark.sql import Column
from pyspark.sql import SparkSession

# noinspection PyUnresolvedReferences
from pyspark.sql.column import _create_column_from_literal
from pyspark.sql.column import _to_java_column
from pyspark.sql.column import _to_seq
from pyspark.sql.functions import countDistinct as count_distinct


def vector_averaging(vecs_col, vec_dim_col):
    sc = SparkContext._active_spark_context
    return Column(
        sc._jvm.org.apache.spark.lightautoml.utils.functions.vector_averaging(
            _to_java_column(vecs_col), _to_java_column(vec_dim_col)
        )
    )


def scalar_averaging(cols):
    sc = SparkContext._active_spark_context
    return Column(sc._jvm.org.apache.spark.lightautoml.utils.functions.scalar_averaging(_to_java_column(cols)))


def get_ctx_for_df(spark: SparkSession):
    # noinspection PyUnresolvedReferences,PyProtectedMember
    return spark._wrapped if parse(version("pyspark")) < parse("3.3.0") else spark


if parse(version("pyspark")) >= parse("3.1.0"):
    from pyspark.ml.functions import array_to_vector
    from pyspark.sql.functions import aggregate
    from pyspark.sql.functions import percentile_approx
    from pyspark.sql.functions import transform
else:

    def _get_lambda_parameters(f):
        import inspect

        signature = inspect.signature(f)
        parameters = signature.parameters.values()

        # We should exclude functions that use
        # variable args and keyword argnames
        # as well as keyword only args
        supported_parameter_types = {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }

        # Validate that
        # function arity is between 1 and 3
        if not (1 <= len(parameters) <= 3):
            raise ValueError(
                "f should take between 1 and 3 arguments, but provided function takes {}".format(len(parameters))
            )

        # and all arguments can be used as positional
        if not all(p.kind in supported_parameter_types for p in parameters):
            raise ValueError("f should use only POSITIONAL or POSITIONAL OR KEYWORD arguments")

        return parameters

    def _unresolved_named_lambda_variable(*name_parts):
        """
        Create `o.a.s.sql.expressions.UnresolvedNamedLambdaVariable`,
        convert it to o.s.sql.Column and wrap in Python `Column`

        Parameters
        ----------
        name_parts : str
        """
        sc = SparkContext._active_spark_context
        name_parts_seq = _to_seq(sc, name_parts)
        expressions = sc._jvm.org.apache.spark.sql.catalyst.expressions
        return Column(sc._jvm.Column(expressions.UnresolvedNamedLambdaVariable(name_parts_seq)))

    def _create_lambda(f):
        """
        Create `o.a.s.sql.expressions.LambdaFunction` corresponding
        to transformation described by f

        :param f: A Python of one of the following forms:
                - (Column) -> Column: ...
                - (Column, Column) -> Column: ...
                - (Column, Column, Column) -> Column: ...
        """
        parameters = _get_lambda_parameters(f)

        sc = SparkContext._active_spark_context
        expressions = sc._jvm.org.apache.spark.sql.catalyst.expressions

        argnames = ["x", "y", "z"]
        args = [_unresolved_named_lambda_variable(arg) for arg in argnames[: len(parameters)]]

        result = f(*args)

        if not isinstance(result, Column):
            raise ValueError("f should return Column, got {}".format(type(result)))

        jexpr = result._jc.expr()
        jargs = _to_seq(sc, [arg._jc.expr() for arg in args])

        return expressions.LambdaFunction(jexpr, jargs, False)

    def _invoke_higher_order_function(name, cols, funs):
        sc = SparkContext._active_spark_context
        expressions = sc._jvm.org.apache.spark.sql.catalyst.expressions
        expr = getattr(expressions, name)

        jcols = [_to_java_column(col).expr() for col in cols]
        jfuns = [_create_lambda(f) for f in funs]

        return Column(sc._jvm.Column(expr(*jcols + jfuns)))

    def array_to_vector(col):
        sc = SparkContext._active_spark_context
        return Column(sc._jvm.org.apache.spark.sql.lightautoml.functions.array_to_vector(_to_java_column(col)))

    def percentile_approx(col, percentage, accuracy=10000):
        sc = SparkContext._active_spark_context

        if isinstance(percentage, (list, tuple)):
            # A local list
            percentage = sc._jvm.functions.array(_to_seq(sc, [_create_column_from_literal(x) for x in percentage]))
        elif isinstance(percentage, Column):
            # Already a Column
            percentage = _to_java_column(percentage)
        else:
            # Probably scalar
            percentage = _create_column_from_literal(percentage)

        accuracy = _to_java_column(accuracy) if isinstance(accuracy, Column) else _create_column_from_literal(accuracy)

        return Column(
            sc._jvm.org.apache.spark.sql.lightautoml.functions.percentile_approx(
                _to_java_column(col), percentage, accuracy
            )
        )

    def aggregate(col, initialValue, merge, finish=None):
        if finish is not None:
            return _invoke_higher_order_function("ArrayAggregate", [col, initialValue], [merge, finish])

        else:
            return _invoke_higher_order_function("ArrayAggregate", [col, initialValue], [merge])

    def transform(col, f):
        return _invoke_higher_order_function("ArrayTransform", [col], [f])


array_to_vector = array_to_vector
count_distinct = count_distinct
percentile_approx = percentile_approx
aggregate = aggregate
transform = transform
