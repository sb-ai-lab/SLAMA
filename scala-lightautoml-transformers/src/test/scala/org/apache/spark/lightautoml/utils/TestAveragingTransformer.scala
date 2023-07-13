package org.apache.spark.lightautoml.utils

import org.apache.spark.internal.Logging
import org.apache.spark.lightautoml.utils.functions.{scalarAveragingUdf, vectorAveragingUdf}
import org.apache.spark.ml.functions.{array_to_vector, vector_to_array}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, DataTypes, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import scala.collection.JavaConverters._

class TestAveragingTransformer extends AnyFunSuite with BeforeAndAfterAll with Logging {
  val num_workers = 3
  val num_cores = 2
  val folds_count = 5

  val spark: SparkSession = SparkSession
          .builder()
          .master(s"local[$num_cores, 1024]")
          .getOrCreate()

  import spark.implicits.StringToColumn

  override protected def afterAll(): Unit = {
    spark.stop()
  }

  test("Vector-based AveragingTransformer") {
    val numPreds = 5
    val dimSize = 3
    val fields = StructField(s"id", DataTypes.IntegerType) :: StructField(s"correct_answer", ArrayType(DataTypes.DoubleType)) :: (0 until numPreds)
            .map(i => StructField(s"pred_$i", ArrayType(DataTypes.DoubleType))).toList
    val schema = StructType(fields)
    val data = List(
      Row(0, Array(0.38, 0.23, 0.39), Array(0.38, 0.23, 0.39), null, null, null, null),
      Row(2, Array(0.38, 0.23, 0.39), null, null, Array(0.38, 0.23, 0.39), null, null),
      Row(3, Array(0.0, 0.0, 0.0), null, null, null, null, null),
      Row(4, Array(0.445, 0.255, 0.3), Array(0.38, 0.23, 0.39), Array(0.21, 0.48, 0.31), Array(0.99, 0.01, 0.0), Array(0.2, 0.3, 0.5), null),
      Row(5, Array(0.4022, 0.3386, 0.2592),Array(0.38, 0.23, 0.39), Array(0.21, 0.48, 0.31), Array(0.99, 0.01, 0.0), Array(0.2, 0.3, 0.5), Array(0.231, 0.673, 0.096)),
    ).asJava

    val cols = (0 until numPreds).map(i => col(s"pred_$i"))
    val vec_cols = array(cols.map(c => when(isnull(c), null).otherwise(array_to_vector(c))): _*)

    val df = spark.createDataFrame(data, schema)
            .select($"id", $"correct_answer", vector_to_array(vectorAveragingUdf(vec_cols, lit(dimSize))).alias("pred"))

    val checks_df = df.select($"id", explode(arrays_zip($"correct_answer", $"pred")).alias("zipped"))
            .select($"id", $"zipped", (abs($"zipped.correct_answer" - $"zipped.pred") < 0.00001).alias("is_correct"))

    assert(checks_df.count() > 0)
    assert(checks_df.count() == checks_df.where($"is_correct").count())
  }

  test("Scalar-based AveragingTransfomer") {
    val numPreds = 5
    val dimSize = 3
    val fields = StructField(s"id", DataTypes.IntegerType) ::
            StructField(s"correct_answer", DataTypes.DoubleType) ::
            (0 until numPreds).map(i => StructField(s"pred_$i", DataTypes.DoubleType)).toList
    val schema = StructType(fields)
    val data = List(
      Row(0, 0.38, 0.38, null, null, null, null),
      Row(2, 0.38, null, null, null, 0.38, null),
      Row(3, 0.0, null, null, null, null, null),
      Row(4, 0.445, 0.38, 0.21, 0.99, 0.2, null),
      Row(5, 0.4022, 0.38, 0.21, 0.99, 0.2, 0.231),
    ).asJava

    val cols = (0 until numPreds).map(i => col(s"pred_$i"))
    val scalar_cols = array(cols: _*)

    val df = spark.createDataFrame(data, schema)
            .select($"id", $"correct_answer", scalarAveragingUdf(scalar_cols).alias("pred"))

    val checks_df = df
            .select($"id", $"correct_answer", $"pred", (abs($"correct_answer" - $"pred") < 0.00001).alias("is_correct"))

    assert(checks_df.count() > 0)
    assert(checks_df.count() == checks_df.where($"is_correct").count())
  }
}
