ThisBuild / organization := "io.github.fonhorst"
ThisBuild / organizationName := "fonhorst"
ThisBuild / organizationHomepage := Some(url("https://github.com/fonhorst"))

ThisBuild / scmInfo := Some(
    ScmInfo(
        url("https://github.com/fonhorst/LightAutoML_Spark"),
        "scm:git@github.com:fonhorst/LightAutoML_Spark.git"
    )
)

ThisBuild / developers := List(
    Developer(
        id    = "fonhorst",
        name  = "Nikolay Butakov",
        email = "alipoov.nb@gmail.com",
        url   = url("https://github.com/fonhorst")
    )
)

ThisBuild / description := "Scala-based implementations of Transformers/Estimators for SLAMA project"
ThisBuild / licenses := List("Apache 2.0" -> new URL("http://www.apache.org/licenses/"))
ThisBuild / homepage := Some(url("https://github.com/fonhorst/LightAutoML_Spark"))

// Remove all additional repository other than Maven Central from POM
ThisBuild / pomIncludeRepository := { _ => false }

ThisBuild / publishTo := {
    val nexus = "https://s01.oss.sonatype.org/"
    if (isSnapshot.value) Some("snapshots" at nexus + "content/repositories/snapshots")
    else Some("releases" at nexus + "service/local/staging/deploy/maven2")
}

ThisBuild / publishMavenStyle := true

ThisBuild / versionScheme := Some("early-semver")
