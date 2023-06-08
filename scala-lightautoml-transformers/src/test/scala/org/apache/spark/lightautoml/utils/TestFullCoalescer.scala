package org.apache.spark.lightautoml.utils

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.must.Matchers.contain
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

import scala.collection.JavaConverters._
import scala.util.Random

class TestFullCoalescer extends AnyFunSuite with BeforeAndAfterEach with Logging {
  val folds_count = 5
  // workers_num, cores_num, slots_num, real_number_of_slots
  val wcs_nums: List[(Int, Int, Int)] = List(
    (3, 2, 3),
    (3, 2, 2),
    (5, 2, 3),
    (5, 1, 6),
    (1, 3, 3),
    (1, 1, 3)
  )

  override protected def afterEach(): Unit = {
    SparkSession.getActiveSession.foreach(spark=> spark.stop())
  }

  wcs_nums.foreach {
    case (num_workers, num_cores, num_slots) =>
      test(s"Coalescers for num_workers=$num_workers, num_cores=$num_cores, num_slots=$num_slots") {
        val spark = SparkSession
                .builder()
                .master(s"local-cluster[$num_workers, $num_cores, 1024]")
                //          .config("spark.jars", "target/scala-2.12/spark-lightautoml_2.12-0.1.1.jar,target/scala-2.12/spark-lightautoml_2.12-0.1.1-tests.jar")
                .config("spark.jars", "target/scala-2.12/spark-lightautoml_2.12-0.1.1.jar")
                .config("spark.default.parallelism", "6")
                .config("spark.sql.shuffle.partitions", "6")
                .config("spark.locality.wait", "15s")
                .getOrCreate()

        import spark.sqlContext.implicits._

        val df = spark
                .sparkContext.parallelize((0 until 100)
                .map(x => (x, Random.nextInt(folds_count)))).toDF("data", "fold")
                .repartition(num_workers * num_cores * 2)
                .cache()
        df.write.mode("overwrite").format("noop").save()

        val all_elements = df.select("data").collect().map(row => row.getAs[Int]("data")).toList

        val (dfs, base_pref_located_df) = SomeFunctions.duplicateOnNumSlotsWithLocationsPreferences(
          df,
          num_slots,
          enforce_division_without_reminder = false
        )

        dfs.size() shouldBe Math.min(num_workers * num_cores, num_slots)

        val durations = new java.util.concurrent.ConcurrentLinkedQueue[Double]()

        val computationsThreads = dfs.asScala.map(df =>{
          val thread = new Thread {
            override def run(): Unit = {
              val t1 = System.nanoTime
              val df_elements = SomeFunctions.test_sleep(df.select("data"))
                      .map(row => row.getAs[Int]("data")).toList
              val duration = (System.nanoTime - t1) / 1e9d
              durations.add(duration)
              df_elements should contain theSameElementsAs all_elements
            }
          }
          thread.start()
          thread
        })

        computationsThreads.foreach(_.join())

        durations.size() shouldBe dfs.size()
        val durs = durations.asScala.toList
        durs.tail.forall(x => math.abs(x - durs.head) <= 1) shouldBe true

        base_pref_located_df.unpersist()

        spark.stop()
      }
  }
}
