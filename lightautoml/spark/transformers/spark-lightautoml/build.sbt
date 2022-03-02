import sbt.Keys.resolvers

name := "spark-lightautoml"

version := "0.1"

scalaVersion := "2.12.12"

//idePackagePrefix := Some("org.apache.spark.ml.feature.lightautoml")

resolvers ++= Seq(
  ("Confluent" at "http://packages.confluent.io/maven")
        .withAllowInsecureProtocol(true)
)

mainClass := Some("org.apache.spark.ml.feature.lightautoml.test_udfs")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.2.0",
  "org.apache.spark" %% "spark-sql" % "3.2.0",
  "org.apache.spark" %% "spark-mllib" % "3.2.0",
  "ml.dmlc" %% "xgboost4j" % "1.5.2",
  "ml.dmlc" %% "xgboost4j-spark" % "1.5.2"
)
