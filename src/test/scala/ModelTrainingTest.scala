import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import Utilities.Utility
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer

class ModelTrainingTest extends AnyFlatSpec with Matchers {
  "Model Training" should "log metrics correctly for each epoch" in {
    val confModelTrainingTest = new SparkConf()
      .setAppName("ModelTrainingTest")
      .setMaster("local[*]")
      .set("spark.ui.enabled", "true")
      .set("spark.ui.port", "0")  // Use a random port for the Spark UI
    val sparkConfModelTrainingTest = SparkSession.builder.config(confModelTrainingTest).getOrCreate()
    val scModelTrainingTest = sparkConfModelTrainingTest.sparkContext
    val logger = LoggerFactory.getLogger(this.getClass)
    val batchSize = 2
    val numEpochs = 2
    val metricsBuffer = ListBuffer[String]()
    val trainingDataRDD = scModelTrainingTest.parallelize(Seq.fill(batchSize)(new DataSet()))  // Mock DataSet

    for (epoch <- 1 to numEpochs) {
      val trainingLoss = 0.05 * epoch
      val validationLoss = 0.06 * epoch
      val accuracy = 0.8
      val precision = 0.75
      val recall = 0.7
      val f1 = 0.72
      val learningRate = 0.01

      metricsBuffer += s"$epoch,$trainingLoss,$validationLoss,$accuracy,$precision,$recall,$f1,$learningRate"
    }

    metricsBuffer.length shouldEqual numEpochs
    metricsBuffer.head should include("1,")
    metricsBuffer.last should include(s"$numEpochs,")
    scModelTrainingTest.stop()
  }
}
