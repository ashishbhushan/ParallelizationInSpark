import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import Utilities.Utility
import org.apache.spark.sql.SparkSession

class EmbeddingsLoadTest extends AnyFlatSpec with Matchers {
  "getEmbeddingsRDD" should "load embeddings and return correct RDD format" in {
    val confEmbeddingsLoadTest = new SparkConf()
      .setAppName("EmbeddingsLoadTest")
      .setMaster("local[*]")
      .set("spark.ui.enabled", "true")
      .set("spark.ui.port", "0")  // Use a random port for the Spark UI
    val sparkConfEmbeddingsLoadTest = SparkSession.builder.config(confEmbeddingsLoadTest).getOrCreate()
    val scEmbeddingsLoadTest = sparkConfEmbeddingsLoadTest.sparkContext
    val path = "src/main/resources/input/embeddings/part-r-00000"  // Path to a small sample embedding file
    val embeddingsRDD = Utility.getEmbeddingsRDD(path, scEmbeddingsLoadTest)
    val firstEmbedding = embeddingsRDD.first()

    firstEmbedding._1 shouldBe a [Int]
    firstEmbedding._2 shouldBe a [Array[Double]]
    scEmbeddingsLoadTest.stop()
  }
}
