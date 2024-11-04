package Utilities

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}


object Utility {
  val windowSize = 10
  val batchSize = 1024
  val numEpochs = 5

  private val loggerUtility: Logger = LoggerFactory.getLogger(this.getClass)

  // function to create a MultiLayerNetwork model
  def createModel(inputSize: Int, lstmLayerSize: Int, outputSize: Int): MultiLayerNetwork = {
    loggerUtility.info(s"Building model with inputSize: $inputSize")
    loggerUtility.info(s"Building model with outputSize: $outputSize")
    val conf = new NeuralNetConfiguration.Builder()
      .updater(new Adam(0.01)) // Set Adam optimizer with a learning rate of 0.01
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(inputSize)
        .nOut(lstmLayerSize)
        .activation(Activation.RELU)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .nIn(lstmLayerSize)
        .nOut(outputSize)
        .activation(Activation.IDENTITY)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    loggerUtility.info("Model initialized successfully")
    model
  }

  // Function to compute positional embedding matrix
  def computePositionalEmbedding(windowSize: Int, embeddingDim: Int): Array[Array[Double]] = {
    val positionalEncoding = Array.ofDim[Double](windowSize, embeddingDim)
    for (pos <- 0 until windowSize; i <- 0 until embeddingDim by 2) {
      val angle = pos / math.pow(10000, 2.0 * i / embeddingDim)
      positionalEncoding(pos)(i) = math.sin(angle)
      if (i + 1 < embeddingDim) positionalEncoding(pos)(i + 1) = math.cos(angle)
    }
    positionalEncoding
  }

  // function to get embeddings in RDD
  def getEmbeddingsRDD(path: String,  context: SparkContext): RDD[(Int, Array[Double])] = {
    val embeddings = context.textFile(path)
      .map { line =>
        val parts = line.split("\t")
        val tokenID = parts(0).toInt
        val embedding = parts(1).split(",").map(_.trim.toDouble)  // Convert entire embedding to Array[Double]
        (tokenID, embedding)
      }
    loggerUtility.info("Embeddings processed to RDD")
    embeddings
  }

  // function to get tokens in RDD
  def getTokensRDD(path: String,  context: SparkContext): RDD[Array[Int]] = {
    val tokens = context.textFile(path)
      .map(line => {
        line.trim.stripPrefix("[").stripSuffix("]").split(",").map(_.toInt)
      })
    loggerUtility.info("Tokens processed to RDD")
    tokens
  }

  // function to get training Dataset
  def getTrainingDataRDD(tokensRDD: RDD[Array[Int]],
                         broadcastTokensWithEmbeddings: Broadcast[Map[Int, Array[Double]]],
                         embeddingDimension: Int,
                         positionalEmbedding: Array[Array[Double]],
                         windowSize: Int): RDD[DataSet] = {
    val dataSet = tokensRDD.flatMap { tokenSeq =>
      tokenSeq.sliding(windowSize).map { window =>
        val positionAwareEmbeddings = window.zipWithIndex.map { case (tokenId, pos) =>
          val tokenEmbedding = broadcastTokensWithEmbeddings.value.getOrElse(tokenId, Array.fill(embeddingDimension)(0.0))
          tokenEmbedding.zip(positionalEmbedding(pos)).map { case (e, p) => e + p }
        }

        // Flatten the embeddings and ensure the array has exactly 250 elements
        val flattenedEmbeddings = positionAwareEmbeddings.flatten
        val featureArray = if (flattenedEmbeddings.length == windowSize*embeddingDimension) {
          flattenedEmbeddings
        } else if (flattenedEmbeddings.length > windowSize*embeddingDimension) {
          flattenedEmbeddings.take(windowSize*embeddingDimension) // Truncate if too long
        } else {
          flattenedEmbeddings ++ Array.fill(windowSize*embeddingDimension - flattenedEmbeddings.length)(0.0) // Pad if too short
        }

        // create features
        val features = Nd4j.create(featureArray).reshape(1, windowSize*embeddingDimension)

        // Label: The last token’s embedding in the window
        val labelArray = broadcastTokensWithEmbeddings.value.getOrElse(window.last, Array.fill(embeddingDimension)(0.0))
        val labels = Nd4j.create(labelArray).reshape(1, embeddingDimension)

        new DataSet(features, labels)
      }
    }
    loggerUtility.info("RDD DataSet processed")
    dataSet
  }

  def getParameterAveragingTM(batchSize: Int): ParameterAveragingTrainingMaster = {
    loggerUtility.info("Training master created")
    new ParameterAveragingTrainingMaster.Builder(batchSize)
      .batchSizePerWorker(batchSize)
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)
      .rddTrainingApproach(RDDTrainingApproach.Export)
      .build()
  }

}
