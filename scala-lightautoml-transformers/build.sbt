import sbt.Keys.resolvers

name := "spark-lightautoml"

version := "0.1.1"

scalaVersion := "2.12.15"

//idePackagePrefix := Some("org.apache.spark.ml.feature.lightautoml")

resolvers ++= Seq(
  ("Confluent" at "http://packages.confluent.io/maven")
        .withAllowInsecureProtocol(true)
)

libraryDependencies ++= Seq(
    "com.microsoft.azure" % "synapseml_2.12" % "0.9.5",
    "org.apache.spark" %% "spark-core" % "3.3.1" % "provided",
    "org.apache.spark" %% "spark-sql" % "3.3.1" % "provided",
    "org.apache.spark" %% "spark-mllib" % "3.3.1" % "provided",
    "org.scalatest" %% "scalatest" % "3.2.14" % Test
)

// uncomment the following lines if you need to build a fat jar
//lazy val app = (project in file("."))
//assemblyMergeStrategy in assembly := {
//    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
//    case x => MergeStrategy.first
//}
//assembly / assemblyJarName := "spark-lightautoml-assembly-fatjar-0.1.1.jar"
