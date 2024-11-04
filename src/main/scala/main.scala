import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.broadcast.Broadcast
import org.apache.hadoop.fs.{FileSystem, Path}
import org.nd4j.evaluation.classification.Evaluation

import java.io.OutputStreamWriter
import scala.collection.mutable.ListBuffer
import java.io.OutputStream
import Utilities.Utility._
import org.slf4j.LoggerFactory

import scala.collection.convert.ImplicitConversions.`iterator asScala`

object main {
  def main(args: Array[String]): Unit = {

    val logger = LoggerFactory.getLogger(this.getClass)

    val environment = if (args.isEmpty) "local" else args(0).toLowerCase
    val inputPath = args(1)
    val outputPath = args(2)

    val conf = new SparkConf()
          .setAppName("CS441HW2")
          .set("spark.storage.blockManagerSlaveTimeoutMs", "500000")  // Default is 120000 (2 min)
          .set("spark.network.timeout", "600s")  // Default is 120s

    val masterUrl = environment match {
      case "spark" =>
        conf.set("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
        "spark://10.0.0.79:7077"
      case "aws" => "yarn"
      case _ => "local[*]"
    }

    conf.setMaster(masterUrl)

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val sc = spark.sparkContext

    val embeddingsPath = inputPath+"/embeddings/part-r-00000"
    val tokensPath = inputPath+"/tokens"
    val modelOutputPath = outputPath+"/model/model.zip"
    val hdfsPath = new Path(modelOutputPath)
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val metricsFilePath = outputPath+"/metrics/metrics.csv"

    // Load embeddings file as RDD[(Int, Array[Double])]
    val embeddingsRDD: RDD[(Int, Array[Double])] = getEmbeddingsRDD(embeddingsPath, sc)

    // Load and parse token data as RDD[Array[Int]]
    val tokensRDD: RDD[Array[Int]] = getTokensRDD(tokensPath, sc)

    // Flatten tokensRDD to have a single RDD of tokens with positional indexing
    val tokensWithPositions: RDD[(Int, Int)] = tokensRDD.flatMap { tokenSeq =>
      tokenSeq.zipWithIndex
    }.zipWithIndex.map { case ((tokenID, _), globalIdx) =>
      (globalIdx.toInt, tokenID) // Position now starts from 0 and increments across all tokens
    }

    val tokensWithEmbeddings: RDD[(Int, Array[Double])] = tokensWithPositions
      .join(embeddingsRDD)
      .map { case (tokenId, (position, embedding)) => (position, embedding) }

    val broadcastTokensWithEmbeddings: Broadcast[Map[Int, Array[Double]]] = sc.broadcast(tokensWithEmbeddings.collectAsMap().toMap)

    // Calculate embedding dimension based on the first element of tokensWithEmbeddings
    val embeddingDimension: Int = tokensWithEmbeddings.first()._2.length

    val positionalEmbedding: Array[Array[Double]] = computePositionalEmbedding(windowSize, embeddingDimension)

    val trainingDataRDD: RDD[DataSet] = getTrainingDataRDD(tokensRDD, broadcastTokensWithEmbeddings,
                                                          embeddingDimension, positionalEmbedding,
                                                          windowSize)

    val trainingJavaRDD: JavaRDD[DataSet] = trainingDataRDD.toJavaRDD()

    val inputSizeRDD = trainingJavaRDD.first().getFeatures.size(1).toInt
    val outputSizeRDD = trainingJavaRDD.first().getLabels.size(1).toInt
    val lstmLayerSizeRDD = 64

    val model: MultiLayerNetwork = createModel(inputSizeRDD, lstmLayerSizeRDD, outputSizeRDD)

    val trainingMaster: ParameterAveragingTrainingMaster = getParameterAveragingTM(batchSize)

    // Spark DL4J Model for distributed training
    val sparkModel: SparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc, model, trainingMaster)

    // Train the model for number for epochs
    logger.info(s"""Starting model training with JavaRDD[DataSet] for epochs: $numEpochs""")
    model.setListeners(new ScoreIterationListener(1))

    val Array(trainingData, pseudoValidationData) = trainingJavaRDD.randomSplit(Array(0.9, 0.1))

    val metricsBuffer = ListBuffer[String]()
    metricsBuffer += "Epoch,EpochDuration(s),AvgTrainingLoss,ValidationLoss,Accuracy,Precision,Recall,F1Score,LearningRate,Executor,TotalMemory(MB),FreeMemory(MB),Partitions".stripMargin
    def logMetrics(epoch: Int, epochDuration: Double, trainingLoss: Double, validationLoss: Double,
                   accuracy: Double, precision: Double, recall: Double, f1Score: Double,
                   learningRate: Double, executor: String,
                   total: Long, free: Long, partition: Int): Unit = {
      metricsBuffer +=
        s"$epoch,$epochDuration,$trainingLoss,$validationLoss,$accuracy,$precision,$recall,$f1Score,$learningRate,$executor,$total,$free,$partition".stripMargin
    }

    for (epoch <- 1 to numEpochs) {
      // Calculate learning rate for the current epoch
      val epochStartTime = System.currentTimeMillis()
      // FIT
      sparkModel.fit(trainingData)

      // 2. Calculate the average loss across the training JavaRDD[DataSet]
      val averageLoss: Double = sparkModel.calculateScore(trainingData, true)
      println(s"Epoch $epoch, Average Training Loss: $averageLoss")

      // 3. Calculate the average loss across the validation JavaRDD[DataSet]
      val validationLoss: Double = sparkModel.calculateScore(pseudoValidationData, true)
      println(s"Epoch $epoch, Validation Loss: $validationLoss")

      // 4. Accuracy calculation
      val evaluation = new Evaluation(outputSizeRDD)
      val broadCastedModel = sc.broadcast(sparkModel.getNetwork)
      // Evaluate model predictions on the validation set
      val predictionsAndLabels: RDD[(Int, Int)] = pseudoValidationData.map { dataSet =>
        val network = broadCastedModel.value
        val predicted = network.output(dataSet.getFeatures).argMax(1).getInt(0)
        val actual = dataSet.getLabels.argMax(1).getInt(0)
        (predicted, actual)
      }
      // Collect and evaluate
      predictionsAndLabels.collect().foreach { case (predicted, actual) =>
        evaluation.eval(actual, predicted)
      }
      val accuracy = evaluation.accuracy()
      val precision = evaluation.precision()
      val recall = evaluation.recall()
      val f1 = evaluation.f1()
      println(s"Accuracy: $accuracy, Validation Precision: $precision, Recall: $recall, F1 Score: $f1")

      // 4. Learning Rate - kept constant at 0.01
      val learningRate: Double = 0.01
      println(s"Epoch $epoch, Learning Rate: $learningRate")

      //5. Gradient Statistics
      // Retrieve and log gradient statistics
//      val gradients = sparkModel.getNetwork.gradient().gradient()
//      val gradientNorm = gradients.norm2Number().doubleValue()
//      val gradientMin = gradients.minNumber().doubleValue()
//      val gradientMax = gradients.maxNumber().doubleValue()
//      println(s"Gradient Norm: $gradientNorm, Min: $gradientMin, Max: $gradientMax")

      //6. Memory Usage
      val memoryStatus = sc.getExecutorMemoryStatus
      // Define separate lists to hold executor, total memory, and free memory values
      // Combine executor names into a single string
      val executors: String = memoryStatus.keys.mkString("|")
      // Sum total memory and free memory across all executors, converting to MB
      val totalMemory: Long = memoryStatus.values.map(_._1).sum / (1024 * 1024)  // in MB
      val freeMemory: Long = memoryStatus.values.map(_._2).sum / (1024 * 1024)   // in MB

      //7. Tracking Shuffle and Data Skew in Spark
      val partitions = trainingData.getNumPartitions
      println(s"Number of partitions in validationData: $partitions")

      // 1. calculate epoch duration
      val epochEndTime = System.currentTimeMillis()
      val epochTimeSeconds = (epochEndTime - epochStartTime)/1000.0
      logger.info(s"Completed epoch $epoch in $epochTimeSeconds ms")

      // Log metrics for this epoch
      logMetrics(epoch,epochTimeSeconds,averageLoss,validationLoss,accuracy,precision,recall,f1,
        learningRate,executors,totalMemory,freeMemory,partitions)
    }
    logger.info("Model training completed")

    // Save metrics once at the end
    def saveMetrics(logPath: String): Unit = {
      val fs = FileSystem.get(sc.hadoopConfiguration)
      val path = new Path(logPath)
      val outputStream = fs.create(path)
      try {
        val writer = new OutputStreamWriter(outputStream, "UTF-8")
        metricsBuffer.foreach(line => writer.write(line + "\n"))
        writer.flush()
      } finally {
        outputStream.close()
      }
    }

    saveMetrics(metricsFilePath)  // Replace with your path

    // Save the trained model
    val outputStream: OutputStream = fs.create(hdfsPath, true)

    // save model to path
    ModelSerializer.writeModel(sparkModel.getNetwork, outputStream, true)
    logger.info(s"""Model training complete and saved to path: $modelOutputPath""")
    outputStream.close()
    fs.close()

    // delete temp files
    trainingMaster.deleteTempFiles(sc)

    // Stop Spark context
    sc.stop()

  }
}