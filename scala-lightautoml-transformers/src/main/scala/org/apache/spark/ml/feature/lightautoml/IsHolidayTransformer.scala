package org.apache.spark.ml.feature.lightautoml

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.NumericAttribute
import org.apache.spark.ml.feature.lightautoml.IsHolidayTransformer.IsHolidayTransformerWriter
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCols}
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.util.VersionUtils.majorMinorVersion

import scala.collection.JavaConverters._
import scala.collection.{Map, mutable}

class IsHolidayTransformer(override val uid: String, private var holidays_dates: Map[String, Set[String]])
        extends Transformer
                with HasInputCols
                with HasOutputCols
                with MLWritable {

  val dt_format = "yyyy-MM-dd"

  def this(uid: String, holidays_dates: java.util.Map[String, java.util.Set[String]]) =
    this(
      uid,
      holidays_dates.asScala.map {case(col_name, mapping) => (col_name, mapping.asScala.toSet)}.toMap
    )

  def setHolidaysDates(holidays_dates: Map[String, Set[String]]): this.type = {
    this.holidays_dates = holidays_dates
    this
  }

  def setInputCols(cols: Array[String]): this.type = set(inputCols, cols)

  def setOutputCols(cols: Array[String]): this.type = set(outputCols, cols)

  def getHolidaysDates: Option[Map[String, Set[String]]] = Some(holidays_dates)

  private def validateAndTransformField(schema: StructType,
                                        inputColName: String,
                                        outputColName: String): StructField = {
    val inputDataType = schema(inputColName).dataType
    require(inputDataType == DateType || inputDataType == TimestampType
            || inputDataType == LongType || inputDataType == IntegerType || inputDataType == StringType,
      s"The input column $inputColName must be of date or timestamp  or long or string type, " +
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

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    transformSchema(dataset.schema)

    val holidaysDatesBcst = spark.sparkContext.broadcast(getHolidaysDates.get)
    val func = udf((col_name: String, date: String) => {
      holidaysDatesBcst.value(col_name)(date)
    })

    val outColumns = getInputCols.zip(getOutputCols).map {
      case (in_col, out_col) =>
        val dt_col = dataset.schema(in_col).dataType match {
          case _: DateType => date_format(to_timestamp(col(in_col)), dt_format)
          case _: TimestampType => date_format(col(in_col), dt_format)
          case _: LongType => date_format(col(in_col).cast(TimestampType), dt_format)
          case _: IntegerType => date_format(col(in_col).cast(TimestampType), dt_format)
          case _: StringType => col(in_col)
        }
        func(lit(in_col).cast(StringType), dt_col).alias(out_col)
    }

    dataset.withColumns(getOutputCols, outColumns)
  }

  override def copy(extra: ParamMap): Transformer = {
    val copied = new IsHolidayTransformer(uid, holidays_dates)
    copyValues(copied, extra)
  }

  override def write: IsHolidayTransformerWriter = new IsHolidayTransformerWriter(this)

  override def toString: String = {
    s"IsHolidayTransformer: uid=$uid, " +
            get(inputCols).map(c => s", numInputCols=${c.length}").getOrElse("") +
            get(outputCols).map(c => s", numOutputCols=${c.length}").getOrElse("")
  }
}


object IsHolidayTransformer extends MLReadable[IsHolidayTransformer] {
  override def read: MLReader[IsHolidayTransformer] = new IsHolidayTransformerReader

  override def load(path: String): IsHolidayTransformer = super.load(path)

  private[IsHolidayTransformer] class IsHolidayTransformerWriter(instance: IsHolidayTransformer)
          extends MLWriter {

    private case class Data(holidays_dates: Map[String, Array[String]])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.getHolidaysDates.get.map{case (col_name, dates) => (col_name, dates.toArray)}.toMap)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private[IsHolidayTransformer] class IsHolidayTransformerReader extends MLReader[IsHolidayTransformer] {

    private val className = classOf[IsHolidayTransformer].getName

    override def load(path: String): IsHolidayTransformer = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString

      // We support loading old `StringIndexerModel` saved by previous Spark versions.
      // Previous model has `labels`, but new model has `labelsArray`.
      val (majorVersion, _) = majorMinorVersion(metadata.sparkVersion)
      if (majorVersion < 3) {
        throw new RuntimeException("Spark version < 3 is not supported")
      }

      val data = sparkSession.read.parquet(dataPath)
              .select("holidays_dates")
              .first()

      val holidays_dates = data.getMap[String, mutable.WrappedArray[String]](0).map{
        case(col_name, warr) => (col_name, warr.toSet)
      }.toMap

      val model = new IsHolidayTransformer(metadata.uid, holidays_dates)
      metadata.getAndSetParams(model)
      model
    }
  }
}

