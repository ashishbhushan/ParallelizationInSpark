import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import Utilities.Utility

class PositionalEmbeddingTest extends AnyFlatSpec with Matchers {
  "computePositionalEmbedding" should "generate positional embeddings with correct dimensions" in {
    val windowSize = 10
    val embeddingDim = 50
    val positionalEmbedding = Utility.computePositionalEmbedding(windowSize, embeddingDim)

    positionalEmbedding.length shouldEqual windowSize
    positionalEmbedding.head.length shouldEqual embeddingDim
  }
}
