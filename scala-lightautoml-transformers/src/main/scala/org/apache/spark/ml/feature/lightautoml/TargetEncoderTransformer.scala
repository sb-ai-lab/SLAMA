package org.apache.spark.ml.feature.lightautoml

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.annotation.Since
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.NumericAttribute
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCols}
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.util.VersionUtils.majorMinorVersion

import scala.collection.{Map, mutable}
import scala.collection.JavaConverters._

// encodings - (column, cat_seq_id -> value)
// oof_encodings - (columns, fold_id -> cat_seq_id -> value)
class TargetEncoderTransformer(override val uid: String,
                               private var enc: TargetEncoderTransformer.Encodings,
                               private var oof_enc: TargetEncoderTransformer.OofEncodings,
                               private var fold_column: String,
                               private var apply_oof: Boolean = true
                              )
        extends Transformer
                with HasInputCols
                with HasOutputCols
                with MLWritable {
//                with DefaultParamsWritable
//                with DefaultParamsReadable[TargetEncoderTransformer] {

  import TargetEncoderTransformer._

  def this(uid: String,
           enc: java.util.Map[String, java.util.List[Double]],
           oof_enc: java.util.Map[String, java.util.List[java.util.List[Double]]],
           fold_column: String, apply_oof: Boolean) =
    this(
      uid,
      enc.asScala.map {case(col_name, mapping) => (col_name, mapping.asScala.toArray)}.toMap,
      oof_enc.asScala.map {case(col_name, mapping) => (col_name, mapping.asScala.map(_.asScala.toArray).toArray)}.toMap,
      fold_column,
      apply_oof
    )

  def setEncodings(enc: Encodings): this.type = {this.enc = enc; this}

  def getEncodings: Option[Encodings] = Some(enc)

  def setOofEncodings(oof_enc: OofEncodings): this.type = {this.oof_enc = oof_enc; this}

  def getOofEncodings: Option[OofEncodings] = Some(oof_enc)

  def setApplyOof(oof: Boolean): this.type = {this.apply_oof = oof; this}

  def getApplyOof: Option[Boolean] = Some(apply_oof)

  def setFoldColumn(col: String): this.type = {this.fold_column = col; this}

  def getFoldColumn: Option[String] = Some(this.fold_column)

  def setInputCols(cols: Array[String]): this.type = set(inputCols, cols)

  def setOutputCols(cols: Array[String]): this.type = set(outputCols, cols)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    transformSchema(dataset.schema)

    val outColumns = getApplyOof match {
      case Some(true) if getOofEncodings.isEmpty =>
        throw new IllegalArgumentException("OofEncodings cannot be unset if applyOof is true")

      case Some(true) if getFoldColumn.isEmpty =>
        throw new IllegalArgumentException("foldCol cannot be unset if applyOof is true")

      case Some(true) =>
        val oofEncodingsBcst = spark.sparkContext.broadcast(getOofEncodings.get)
        val func = udf((col_name: String, fold: Integer, cat: Integer) => {
          oofEncodingsBcst.value(col_name)(fold)(cat)
        })
        setApplyOof(false)
        val outs = getInputCols.zip(getOutputCols).map{
          case (in_col, out_col) => func(
            lit(in_col).cast(StringType),
            col(getFoldColumn.get).cast(IntegerType),
            col(in_col).cast(IntegerType)
          ).alias(out_col)
        }
        outs

      case Some(false) if getEncodings.isEmpty =>
        throw new IllegalArgumentException("Encodings cannot be unset if applyOof is false")

      case Some(false) =>
        val encodingsBcst = spark.sparkContext.broadcast(getEncodings.get)
        val func = udf((col_name: String, cat: Integer) => {
          encodingsBcst.value(col_name)(cat)
        })
        getInputCols.zip(getOutputCols).map {
          case (in_col, out_col) => func(
            lit(in_col),
            col(in_col).cast(IntegerType)
          ).alias(out_col)
        }

      case None =>
        throw new IllegalArgumentException("applyOof cannot be None")
    }

    dataset.withColumns(getOutputCols, outColumns)
  }

  override def copy(extra: ParamMap): Transformer = {
    val copied = new TargetEncoderTransformer(uid, enc, oof_enc, fold_column)
    copyValues(copied, extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def write: TargetEncoderTransformerWriter = new TargetEncoderTransformerWriter(this)

  override def toString: String = {
    s"TargetEncoderTransformer: uid=$uid, " +
            getApplyOof.map(t => s", applyOf=$t").getOrElse("") +
            get(inputCols).map(c => s", numInputCols=${c.length}").getOrElse("") +
            get(outputCols).map(c => s", numOutputCols=${c.length}").getOrElse("")
  }

  private def validateAndTransformField(schema: StructType,
                                        inputColName: String,
                                        outputColName: String): StructField = {
    val inputDataType = schema(inputColName).dataType
    require(inputDataType == IntegerType || inputDataType == ShortType || inputDataType == LongType,
      s"The input column $inputColName must be integer type or short type, " +
              s"but got $inputDataType.")
    require(schema.fields.forall(_.name != outputColName),
      s"Output column $outputColName already exists.")
    NumericAttribute.defaultAttr.withName(outputColName).toStructField()
  }

  private def validateAndTransformSchema(schema: StructType,
                                         skipNonExistsCol: Boolean = false): StructType = {
    val inputColNames = getInputCols
    val outputColNames = getOutputCols

    require(outputColNames.distinct.length == outputColNames.length,
      s"Output columns should not be duplicate.")

    val outputFields = inputColNames.zip(outputColNames).flatMap {
      case (inputColName, outputColName) =>
        schema.fieldNames.contains(inputColName) match {
          case true => Some(validateAndTransformField(schema, inputColName, outputColName))
          case false if skipNonExistsCol => None
          case _ => throw new SparkException(s"Input column $inputColName does not exist.")
        }
    }
    StructType(schema.fields ++ outputFields)
  }
}

object TargetEncoderTransformer extends MLReadable[TargetEncoderTransformer] {
  type Encodings = Map[String, Array[Double]]
  type OofEncodings = Map[String, Array[Array[Double]]]

  override def read: MLReader[TargetEncoderTransformer] = new TargetEncoderTransformerReader

  override def load(path: String): TargetEncoderTransformer = super.load(path)

  private[TargetEncoderTransformer] class TargetEncoderTransformerWriter(instance: TargetEncoderTransformer)
          extends MLWriter {

    private case class Data(encodings: Encodings, oofEncodings: OofEncodings, applyOof: Boolean, foldColumn: String)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(
        instance.getEncodings.get,
        instance.getOofEncodings.get,
        instance.getApplyOof.get,
        instance.getFoldColumn.get
      )
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private[TargetEncoderTransformer] class TargetEncoderTransformerReader extends MLReader[TargetEncoderTransformer] {

    private val className = classOf[TargetEncoderTransformer].getName

    override def load(path: String): TargetEncoderTransformer = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString

      // We support loading old `StringIndexerModel` saved by previous Spark versions.
      // Previous model has `labels`, but new model has `labelsArray`.
      val (majorVersion, _) = majorMinorVersion(metadata.sparkVersion)
      if (majorVersion < 3) {
        throw new RuntimeException("Spark version < 3 is not supported")
      }

      val data = sparkSession.read.parquet(dataPath)
              .select("encodings", "oofEncodings", "applyOof", "foldColumn")
              .first()

      val enc = data.getMap[String, mutable.WrappedArray[Double]](0).map{
        case(col_name, warr) => (col_name, warr.toArray)
      }.toMap
      val oofEnc = data.getMap[String, mutable.WrappedArray[mutable.WrappedArray[Double]]](1).map{
        case(col_name, warr) => (col_name, warr.map(_.toArray).toArray)
      }.toMap
      val applyOof = data.getBoolean(2)
      val foldColumns = data.getString(3)

      val model = new TargetEncoderTransformer(metadata.uid, enc, oofEnc, foldColumns, applyOof)
      metadata.getAndSetParams(model)
      model
    }
  }
}
