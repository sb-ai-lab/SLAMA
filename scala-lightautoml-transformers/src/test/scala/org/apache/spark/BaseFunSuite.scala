package org.apache.spark

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import java.io.File
import scala.reflect.io.Directory

abstract class BaseFunSuite extends AnyFunSuite with BeforeAndAfterAll with Logging {
  // scalastyle:on

  // Initialize the logger forcibly to let the logger log timestamp
  // based on the local time zone depending on environments.
  // The default time zone will be set to America/Los_Angeles later
  // so this initialization is necessary here.
  log

  var spark: SparkSession = _

  val workdir: String = "/tmp/test_transformers"

  protected override def beforeAll(): Unit = {
    spark =
      SparkSession
              .builder()
              .appName("test")
              .master("local[1]")
              .getOrCreate()


    val dir = new Directory(new File(workdir))
    dir.deleteRecursively()
    dir.createDirectory()
  }

  protected override def afterAll(): Unit = {
    spark.stop()

    val dir = new Directory(new File(workdir))
    dir.deleteRecursively()
  }
}
