import sbtassembly.MergeStrategy

ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

libraryDependencies ++= Seq(
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
  "org.deeplearning4j" % "dl4j-spark_2.12" % "1.0.0-M2.1",
  "org.apache.spark" % "spark-core_2.12" % "3.5.3",
  "org.apache.spark" % "spark-mllib_2.12" % "3.5.3",
  "org.slf4j" % "slf4j-api" % "2.0.12",
  "ch.qos.logback" % "logback-classic" % "1.5.6",
  "org.scalatest" %% "scalatest" % "3.2.19" % Test,
  "org.scalatestplus" %% "scalacheck-1-15" % "3.2.11.0" % Test
)

// Enable using ScalaCheck with ScalaTest
testFrameworks += new TestFramework("org.scalatest.tools.Framework")
// Set parallel execution for tests to false using the slash syntax
Test / parallelExecution := false

ThisBuild / assemblyMergeStrategy := {
  case x if Assembly.isConfigFile(x) =>
    MergeStrategy.concat
  case PathList(ps @ _*) if Assembly.isReadme(ps.last) || Assembly.isLicenseFile(ps.last) =>
    MergeStrategy.rename
  case PathList("META-INF", "services", "org.apache.hadoop.fs.FileSystem") =>
    MergeStrategy.filterDistinctLines
  case PathList("META-INF", xs @ _*) =>

    (xs map {_.toLowerCase}) match {
      case ("manifest.mf" :: Nil) | ("index.list" :: Nil) | ("dependencies" :: Nil) =>
        MergeStrategy.discard
      case ps @ (x :: xs) if ps.last.endsWith(".sf") || ps.last.endsWith(".dsa") =>
        MergeStrategy.discard
      case "plexus" :: xs =>
        MergeStrategy.discard
      case "services" :: xs =>
        MergeStrategy.filterDistinctLines
      case ("spring.schemas" :: Nil) | ("spring.handlers" :: Nil) =>
        MergeStrategy.filterDistinctLines
      case _ => MergeStrategy.first
    }
  case _ => MergeStrategy.first
}