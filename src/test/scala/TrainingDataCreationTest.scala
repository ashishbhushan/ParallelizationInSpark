import org.nd4j.linalg.dataset.DataSet
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import Utilities.Utility
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

class TrainingDataCreationTest extends AnyFlatSpec with Matchers {
  "getTrainingDataRDD" should "create RDD[DataSet] with proper feature and label shapes" in {
    val confTrainingDataCreationTest = new SparkConf()
      .setAppName("TrainingDataCreationTest")
      .setMaster("local[*]")
      .set("spark.ui.enabled", "true")
      .set("spark.ui.port", "0")  // Use a random port for the Spark UI
    val sparkTrainingDataCreationTest = SparkSession.builder.config(confTrainingDataCreationTest).getOrCreate()
    val scTrainingDataCreationTest = sparkTrainingDataCreationTest.sparkContext
    val tokensRDD = Utility.getTokensRDD("src/main/resources/input/tokens/token_1.txt", scTrainingDataCreationTest)
    val broadcastTokensWithEmbeddings = scTrainingDataCreationTest.broadcast(Map(1 -> Array.fill(50)(0.5)))  // Mock embeddings
    val windowSize = 10
    val embeddingDimension = 50
    val positionalEmbedding = Utility.computePositionalEmbedding(windowSize, embeddingDimension)

    val trainingDataRDD = Utility.getTrainingDataRDD(tokensRDD, broadcastTokensWithEmbeddings, embeddingDimension, positionalEmbedding, windowSize)
    val firstDataSet: DataSet = trainingDataRDD.first()

    firstDataSet.getFeatures.shape() shouldEqual Array(1, windowSize * embeddingDimension)
    firstDataSet.getLabels.shape() shouldEqual Array(1, embeddingDimension)
    scTrainingDataCreationTest.stop()
  }
}
