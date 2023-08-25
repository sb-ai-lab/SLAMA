ThisBuild / organization := "io.github.sb-ai-lab"
ThisBuild / organizationName := "sb-ai-lab"
ThisBuild / organizationHomepage := Some(url("https://github.com/sb-ai-lab/"))

ThisBuild / scmInfo := Some(
    ScmInfo(
        url("https://github.com/sb-ai-lab/SLAMA"),
        "scm:git@github.com:sb-ai-lab/SLAMA.git"
    )
)

ThisBuild / developers := List(
    Developer(
        id    = "alexmryzhkov",
        name  = "Alexander Ryzhkov",
        email = "alexmryzhkov@gmail.com",
        url   = url("https://github.com/alexmryzhkov")
    ),
    Developer(
        id    = "btbpanda",
        name  = "Anton Vakhrushev",
        email = "btbpanda@gmail.com",
        url   = url("https://github.com/btbpanda")
    ),
    Developer(
        id    = "DESimakov",
        name  = "Dmitrii Simakov",
        email = "dmitryevsimakov@gmail.com",
        url   = url("https://github.com/DESimakov")
    ),
    Developer(
        id    = "dev-rinchin",
        name  = "Rinchin Damdinov",
        email = "damdinovr@gmail.com",
        url   = url("https://github.com/dev-rinchin")
    ),
    Developer(
        id    = "Cybsloth",
        name  = "Alexander Kirilin",
        email = "adkirilin@gmail.com",
        url   = url("https://github.com/Cybsloth")
    ),
    Developer(
        id    = "VaBun",
        name  = "Vasilii Bunakov",
        email = "va.bunakov@gmail.com",
        url   = url("https://github.com/VaBun")
    ),
    Developer(
        id    = "fonhorst",
        name  = "Nikolay Butakov",
        email = "alipoov.nb@gmail.com",
        url   = url("https://github.com/fonhorst")
    ),
    Developer(
        id    = "netang",
        name  = "Azamat Gainetdinov",
        email = "Mr.g.azamat@gmail.com",
        url   = url("https://github.com/netang")
    ),
    Developer(
        id    = "se-teryoshkin",
        name  = "Sergey Teryoshkin",
        email = "se.teryoshkin@gmail.com",
        url   = url("https://github.com/se-teryoshkin")
    ),
    Developer(
        id    = "dnasonov",
        name  = "Denis Nasonov",
        email = "denis.nasonov@gmail.com",
        url   = url("https://github.com/dnasonov")
    )
)

ThisBuild / description := "Scala-based implementations of Transformers/Estimators for SLAMA project"
ThisBuild / licenses := List("Apache 2.0" -> new URL("http://www.apache.org/licenses/"))
ThisBuild / homepage := Some(url("https://github.com/sb-ai-lab/SLAMA"))

// Remove all additional repository other than Maven Central from POM
ThisBuild / pomIncludeRepository := { _ => false }

ThisBuild / publishTo := {
    val nexus = "https://s01.oss.sonatype.org/"
    if (isSnapshot.value) Some("snapshots" at nexus + "content/repositories/snapshots")
    else Some("releases" at nexus + "service/local/staging/deploy/maven2")
}

ThisBuild / publishMavenStyle := true

ThisBuild / versionScheme := Some("early-semver")
