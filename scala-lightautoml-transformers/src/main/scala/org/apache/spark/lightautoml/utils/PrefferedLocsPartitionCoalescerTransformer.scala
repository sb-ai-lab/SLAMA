package org.apache.spark.lightautoml.utils

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.StructType
import scala.collection.JavaConverters._

class PrefferedLocsPartitionCoalescerTransformer(override val uid: String,
                                                 val prefLocs: List[String],
                                                 val do_shuffle: Boolean) extends Transformer  {

  def this(uid: String, prefLocs: java.util.List[String], do_shuffle: Boolean) = this(uid, prefLocs.asScala.toList, do_shuffle)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark = SparkSession.active
    val ds = dataset.asInstanceOf[Dataset[Row]]
    val master = spark.sparkContext.master

    val foundCoresNum = spark.conf.getOption("spark.executor.cores") match {
      case Some(cores_option) => Some(cores_option.toInt)
      case None => if (master.startsWith("local-cluster")){
        val cores = master.slice("local-cluster[".length, master.length - 1).split(',')(1).trim.toInt
        Some(cores)
      } else if (master.startsWith("local")) {
        val num_cores = master.slice("local[".length, master.length - 1)
        val cores = if (num_cores == "*") { java.lang.Runtime.getRuntime.availableProcessors } else { num_cores.toInt }
        Some(cores)
      } else {
        None
      }
    }

    assert(foundCoresNum.nonEmpty, "Cannot find number of used cores per executor")

    val execCores = foundCoresNum.get

    val coalesced_rdd = ds.rdd.coalesce(
      numPartitions = execCores * prefLocs.size,
      shuffle = do_shuffle,
      partitionCoalescer = Some(new PrefferedLocsPartitionCoalescer(prefLocs))
    )

    spark.createDataFrame(coalesced_rdd, schema = dataset.schema)
  }

  override def copy(extra: ParamMap): Transformer = new PrefferedLocsPartitionCoalescerTransformer(uid, prefLocs, do_shuffle)

  override def transformSchema(schema: StructType): StructType = schema.copy()
}


object SomeFunctions {
  def func[T](x: T): T = x

  def executors(): java.util.List[java.lang.String] = {
    import scala.collection.JavaConverters._
    if (SparkSession.active.sparkContext.master.startsWith("local[")) {
      SparkSession.active.sparkContext.env.blockManager.master.getMemoryStatus
              .map { case (blockManagerId, _) => blockManagerId }
              .map { executor => s"executor_${executor.host}_${executor.executorId}" }
              .toList.asJava
    } else {
      SparkSession.active.sparkContext.env.blockManager.master.getMemoryStatus
              .map { case (blockManagerId, _) => blockManagerId }
              .filter(_.executorId != "driver")
              .map { executor => s"executor_${executor.host}_${executor.executorId}" }
              .toList.asJava
    }
  }
}