package org.apache.spark.lightautoml.utils

import org.apache.spark.Partitioner
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.{PartitionPruningRDD, ShuffledRDD}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import java.util
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

class TrivialPartitioner(override val numPartitions: Int) extends Partitioner {
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
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

  def test_func(df: DataFrame): Long = {
    df.rdd.barrier().mapPartitions(SomeFunctions.func).count()
  }

  def test_sleep(df: DataFrame, sleep_millis: Int = 5000 ): Array[Row] = {
    df.rdd.mapPartitions(x => {Thread.sleep(sleep_millis); x}).collect()
  }

  /**
   * Makes numSlots copies of dataset and produce a list of dataframes where each one is a copy of the initial dataset.
   * Every copy is coalesced to a number of executors by setting appropriate Preffered Locations.
   * Subsequent map and aggregate operations should happen only on a subset of executors matched with an output dataframe.
   * Be aware that:
   *    1. There may be some unused cores if number of cores x number of executors cannot be divided
   *      by number of slots without remainder
   *    2. Number of slots will be reduced down to number of cores x number of executors if number of slots is greater
   *    3. Enabling enforce_division_without_reminder will lead to an exception
   *      if  number of cores x number of executors cannot be divided by number of slots without remainder
   * */
  def duplicateOnNumSlotsWithLocationsPreferences(df: DataFrame,
                                                  numSlots: Int,
                                                  materialize_base_rdd: Boolean = true,
                                                  enforce_division_without_reminder: Boolean = true):
  (java.util.List[DataFrame], DataFrame) = {
    // prepare and identify params for slots
    val spark = SparkSession.active
    val master = spark.sparkContext.master
    val execs = SomeFunctions.executors()
    val numExecs = execs.size()
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

    val numPartitions = numExecs * foundCoresNum.get

    if (enforce_division_without_reminder) {
      assert(numPartitions % numSlots == 0,
        s"Resulting num partitions should be exactly dividable by num slots: $numPartitions % $numSlots != 0")
      assert(numExecs % numSlots == 0,
        s"Resulting num executors should be exactly dividable by num slots: $numExecs % $numSlots != 0")
    }

    val realNumSlots = math.min(numSlots, numPartitions)

    val partitionsPerSlot = numPartitions / realNumSlots
    val prefLocsForAllPartitions = execs.asScala.flatMap(e_id => (0 until foundCoresNum.get).map(_ => e_id)).toList

    val partition_id_col = "__partition_id"
    // prepare the initial dataset by duplicating its content and assigning partition_id for a specific duplicated rows
    val duplicated_df = df
            .withColumn(
              partition_id_col,
              explode(array((0 until realNumSlots).map(x => lit(x)):_*))
            )
            .withColumn(
              partition_id_col,
              col(partition_id_col) * lit(partitionsPerSlot)
                      + (lit(partitionsPerSlot) * rand(seed = 42)).cast("int")
            )

    // repartition the duplicated dataset to force all desired copies into specific subsets of partitions
    // should work even with standard HashPartitioner
    val new_rdd_2 = new ShuffledRDD[Int, Row, Row](
      duplicated_df.rdd.map(row => (row.getInt(row.fieldIndex(partition_id_col)), row)),
      new TrivialPartitioner(numPartitions)
    ).map(x => x._2)

    val new_rdd_df = spark.createDataFrame(new_rdd_2.coalesce(
      numPartitions,
      shuffle = false,
      partitionCoalescer = Some(new PrefferedLocsPartitionCoalescer(prefLocsForAllPartitions))
    ), duplicated_df.schema)

    // not sure if it is needed or not to perform all operation in parallel
    val copies_rdd_df = if (materialize_base_rdd) {
      val cached_rdd_df = new_rdd_df.cache()
      cached_rdd_df.write.mode("overwrite").format("noop").save()
      cached_rdd_df
    } else {
      new_rdd_df
    }

    // select subsets of partitions that contains independent copies of the initial dataset
    // assign it preferred locations and convert the resulting rdds into DataFrames
    // We need to localCheckpoint these datasets because downstream SynapseML lightGBM
    // cannot work with PrunningPartitionRDD. Precise error:
    //    org.apache.spark.scheduler.BarrierJobUnsupportedRDDChainException: [SPARK-24820][SPARK-24821]:
    //    Barrier execution mode does not allow the following pattern of RDD chain within a barrier stage:
    //    1. Ancestor RDDs that have different number of partitions from the resulting RDD
    //      (e.g. union()/coalesce()/first()/take()/PartitionPruningRDD).
    //      A workaround for first()/take() can be barrierRdd.collect().head (scala) or barrierRdd.collect()[0] (python)
    //    2. An RDD that depends on multiple barrier RDDs (e.g. barrierRdd1.zip(barrierRdd2)).
    //      at org.apache.spark.scheduler.DAGScheduler.checkBarrierStageWithRDDChainPattern(DAGScheduler.scala:447)
    //      at org.apache.spark.scheduler.DAGScheduler.createResultStage(DAGScheduler.scala:590)
    // In the same time, we cannot skip barrier stage in SynapseML LightGBM,
    // because it would lead to applying coalesce inside the lightgbm and will lead to breaking
    // of preffered locations set earlier
    val prefLocsDfs = new util.ArrayList[DataFrame]()
    val threads = (0 until realNumSlots)
            .map (slotId => new PartitionPruningRDD(copies_rdd_df.rdd, x => x / partitionsPerSlot == slotId))
            .map {
              rdd =>
                val thread = new Thread {
                  override def run(): Unit = {
                    val df = spark.createDataFrame(
                      rdd,
                      schema = duplicated_df.schema
                    ).drop(partition_id_col).localCheckpoint(true)
                    prefLocsDfs.add(df)
                  }
                }
                thread.start()
                thread
            }
            .toList

    threads.foreach(_.join())

    copies_rdd_df.unpersist()

    (prefLocsDfs, copies_rdd_df)
  }
}