package org.apache.spark.lightautoml.utils

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

class TestPrefferedLocsPartitionCoalescer extends AnyFunSuite with BeforeAndAfterAll with Logging {
  val num_workers = 3
  val num_cores = 2
  val folds_count = 5

  val spark: SparkSession = SparkSession
          .builder()
          .master(s"local-cluster[$num_workers, $num_cores, 1024]")
          .config("spark.jars", "target/scala-2.12/spark-lightautoml_2.12-0.1.1.jar")
          .config("spark.default.parallelism", "6")
          .config("spark.sql.shuffle.partitions", "6")
          .config("spark.locality.wait", "15s")
          .getOrCreate()

  override protected def afterAll(): Unit = {
    spark.stop()
  }

  test("Coalescers") {
    import spark.sqlContext.implicits._
    val df = spark
            .sparkContext.parallelize((0 until 5)
            .map(x => (x, Random.nextInt(folds_count)))).toDF("data", "fold")
            .repartition(num_workers * num_cores * 2)
            .cache()
    df.write.mode("overwrite").format("noop").save()

    val executor = spark.sparkContext.env.blockManager.master.getMemoryStatus
            .map { case (blockManagerId, _) => blockManagerId}
            .filter(_.executorId != "driver")
            .head

    val prefLocs = List(s"executor_${executor.host}_1", s"executor_${executor.host}_2")

    val coalescerTransformer = new PrefferedLocsPartitionCoalescerTransformer(uid = "some uid", prefLocs, true)
    var coalesced_df = coalescerTransformer.transform(df)

    coalesced_df = coalesced_df.cache()
    coalesced_df.write.mode("overwrite").format("noop").save()

    coalesced_df.count()

    SomeFunctions.test_func(coalesced_df)
  }
}
