package org.apache.spark.ml.feature.lightautoml

import org.apache.spark.BaseFunSuite
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.matchers.should.Matchers._

import scala.collection.JavaConverters._

class TestTransformers extends BaseFunSuite {

  test("Smoke LAMLStringIndexer test") {
    val file = "resources/data.json"
    val testFile = "resources/test_data.json"


    val df = spark.read.json(file).cache()
    val testDf = spark.read.json(testFile).cache()
    val cnt = df.count()

    println("-- Source --")
    df.show(100)

    val startTime = System.currentTimeMillis()

    val lamaIndexer = new LAMLStringIndexer()
            .setMinFreq(Array(1))
            .setFreqLabel(true)
            .setDefaultValue(1.0F)
            .setInputCols(Array("value"))
            .setOutputCols(Array("index"))
            .setHandleInvalid("keep")

    println(lamaIndexer.uid)

    val lamaModel = lamaIndexer.fit(df)
    val lamaTestIndexed = lamaModel.transform(testDf)

    println("-- Lama Indexed --")
    lamaTestIndexed.show(100)

    val endTime = System.currentTimeMillis()

    println(s"Duration = ${(endTime - startTime) / 1000D} seconds")
    println(s"Size: $cnt")

    lamaModel.write.overwrite().save("/tmp/LAMLStringIndexerModel")
    val pipelineModel = LAMLStringIndexerModel.load("/tmp/LAMLStringIndexerModel")
    pipelineModel.transform(testDf)
  }

  test("Smoke test of Target Encoder transformer") {
    val in_cols = Seq("a", "b", "c").toArray
    val out_cols = in_cols.map(x => s"te_$x")

    val enc = Map(
      ("a", Array(0.0, -1.0, -2.0, -3.0, -4.0)),
      ("b", Array(0.0, -1.0, -2.0)),
      ("c", Array(0.0, -1.0, -2.0, -3.0))
    )

    val oof_enc = Map(
      ("a", Array(
        Array(0.0, 10.0, 20.0, 30.0, 40.0),
        Array(0.0, 11.0, 12.0, 13.0, 14.0),
        Array(0.0, 21.0, 22.0, 23.0, 24.0)
      )),
      ("b", Array(
        Array(0.0, 10.0, 20.0),
        Array(0.0, 11.0, 12.0),
        Array(0.0, 21.0, 22.0))
      ),
      ("c", Array(
        Array(0.0, 10.0, 20.0, 30.0),
        Array(0.0, 11.0, 12.0, 13.0),
        Array(0.0, 21.0, 22.0, 23.0)
      ))
    )

    val fold_column = "fold"
    val id_column = "id"

    // id, fold_column, some_other_col, a, b, c
    val data = Seq(
      Row(0, 0, 42, 1, 1, 1),
      Row(1, 0, 43, 2, 1, 3),
      Row(2, 1, 44, 1, 2, 3),
      Row(3, 1, 45, 1, 2, 2),
      Row(4, 2, 46, 3, 1, 1),
      Row(5, 2, 47, 4, 1, 2),
    ).toList.asJava

    val result_enc = Seq(
      Row(0, 0, 42, 1, 1, 1, -1.0, -1.0, -1.0),
      Row(1, 0, 43, 2, 1, 3, -2.0, -1.0, -3.0),
      Row(2, 1, 44, 1, 2, 3, -1.0, -2.0, -3.0),
      Row(3, 1, 45, 1, 2, 2, -1.0, -2.0, -2.0),
      Row(4, 2, 46, 3, 1, 1, -3.0, -1.0, -1.0),
      Row(5, 2, 47, 4, 1, 2, -4.0, -1.0, -2.0),
    )

    val result_oof_enc = Seq(
      Row(0, 0, 42, 1, 1, 1, 10.0, 10.0, 10.0),
      Row(1, 0, 43, 2, 1, 3, 20.0, 10.0, 30.0),
      Row(2, 1, 44, 1, 2, 3, 11.0, 12.0, 13.0),
      Row(3, 1, 45, 1, 2, 2, 11.0, 12.0, 12.0),
      Row(4, 2, 46, 3, 1, 1, 23.0, 21.0, 21.0),
      Row(5, 2, 47, 4, 1, 2, 24.0, 21.0, 22.0),
    )

    val schema = StructType(
      Array(
        StructField(id_column, IntegerType),
        StructField(fold_column, IntegerType),
        StructField("some_other_col", IntegerType)
      )
      ++ in_cols.map(col => StructField(col, IntegerType))
    )

    def checkResult(tdf: DataFrame, df: DataFrame, target_data: Seq[Row]): Unit = {
      tdf.columns should contain allElementsOf df.columns
      tdf.columns should contain allElementsOf out_cols
      out_cols.foreach(col => tdf.schema(col).dataType shouldBe a [DoubleType])

      val resul_rows = tdf.orderBy(col(id_column)).collect()
      resul_rows.zip(target_data).foreach {
        case (row, target) => row.toSeq should equal (target.toSeq)
      }
    }

    val df = spark.createDataFrame(data, schema)

    val te = new TargetEncoderTransformer("te_tr", enc, oof_enc, fold_column)
            .setInputCols(in_cols)
            .setOutputCols(out_cols)

    checkResult(te.transform(df), df, result_oof_enc)
    checkResult(te.transform(df), df, result_enc)
    checkResult(te.transform(df), df, result_enc)

    val path = s"$workdir/target_encoder.transformer"
    te.save(path)

    val loaded_te = TargetEncoderTransformer.load(path)
    checkResult(loaded_te.transform(df), df, result_enc)
    checkResult(loaded_te.transform(df), df, result_enc)
  }
}
