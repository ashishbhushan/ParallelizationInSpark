import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import Utilities.Utility
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

class TokensLoadTest extends AnyFlatSpec with Matchers {
  "getTokensRDD" should "load tokens and return correct RDD format" in {
    val confTokensLoadTest = new SparkConf()
      .setAppName("TokensLoadTest")
      .setMaster("local[*]")
      .set("spark.ui.enabled", "true")
      .set("spark.ui.port", "0")  // Use a random port for the Spark UI
    val sparkConfTokensLoadTest = SparkSession.builder.config(confTokensLoadTest).getOrCreate()
    val scTokensLoadTest = sparkConfTokensLoadTest.sparkContext
    val path = "src/main/resources/input/tokens/token_1.txt"  // Path to a small sample tokens file
    val tokensRDD = Utility.getTokensRDD(path, scTokensLoadTest)
    val firstTokenArray = tokensRDD.first()

    firstTokenArray shouldBe a [Array[Int]]
    firstTokenArray.length should be > 0
    scTokensLoadTest.stop()
  }
}
