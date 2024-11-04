# Implementing Parallelization using Spark

### Author: Ashish Bhushan
### Email: abhus@uic.edu
### UIN: 654108403

## Introduction

Homework Assignment 2 for CS441 centers on harnessing Spark’s distributed computing capabilities to build parallelization from the ground up using cloud-based technologies. In this phase, we will implement positional embeddings, then create, train, and save a SparkDl4jMultiLayer model with the generated embeddings, leveraging Spark’s distributed processing for scalability. We’ll also gather detailed training statistics throughout. After local testing, the program will be deployed and executed on Amazon Elastic MapReduce (EMR) for full-scale distributed processing.


Video Link:

The video explains the deployment of Spark application in the AWS EMR Cluster and the project structure


### Environment
   ```bash
   OS: Windows 11
   
   IDE: IntelliJ IDEA 2024.2.3 (Ultimate Edition)
   
   SCALA Version: 2.12.18
   
   SBT Version: 1.10.4
   
   Spark Version: 3.5.3
   
   Java Version: 11.0.24
   ```


### Running the project

1) **Clone this repository**

   ```bash
   https://github.com/ashishbhushan/CS441HW2.git
   ```

2) **Navigate to the Project:**

   ```bash
   cd CS441HW1
   ```

3) **Open the project in IntelliJ**


   [How to Open a Project in IntelliJ](https://www.jetbrains.com/help/idea/import-project-or-module-wizard.html#open-project)


4) **Run the project via Intellij**

## Follow these steps to execute the project:

Input Data: Ensure you have the output from the previous homework (embeddings and sharded tokens file)

embeddings:https://github.com/ashishbhushan/WordSimilarityMR/blob/master/src/main/resources/output/mapRedEmbeddingOut/part-r-00000

tokens:https://github.com/ashishbhushan/WordSimilarityMR/tree/master/src/main/resources/output/tokens (copy the entire directory)

Both the input datasets should be copied and pasted inside respective directories in src/main/resources/input (embeddings and tokens)

![image](https://github.com/user-attachments/assets/b558911c-9d97-4129-adfe-f5d48dcb65cf)

Output Directory: Ensure output directory is also created in src/main/resources/ with subdirectories metrics (to store metrics.csv) and model (to store model.csv)

![image](https://github.com/user-attachments/assets/2031fc8d-c6e5-4e60-a096-b0dd16b415bb)

Go to Run > Edit Configurations...
Setup the command line using intelliJ configuration if required as shown below. Please do make sure to set shorten command line parameter to jar manifest. Enter input arguments as well.
Input args:
1. local/spark/aws (choose local if running locally via intelliJ, aws for AWS EMR, spark for spark)
2. input path: src/main/resources/input
3. output path: src/main/resources/output

![image](https://github.com/user-attachments/assets/f3b8bed5-499f-459b-a7b1-81549547af0d)

Run the project via the main function - main.scala


```
## Project Structure

The project comprises the following key components:

- Positional Embedding Generation: Embeddings are generated based on the order of the sentences for contextual purposes and generation of Target Embeddings using Spark's parallelization capabilities.

- Training and Storing of Model: A SparkDl4jMultiLayer neural network is created and trained based on the positional embeddings and is stored using Spark's parallelization capabilities.

- Generation of Metrics: The metrics of the model are generated and stored in a file called metrics.csv (some of the metrics are visible in the logs of the Spark runtime)

## Prerequisites

Before starting the project, ensure that you have the necessary tools and accounts set up:

1. **Hadoop**: Set up Hadoop on your local machine or cluster.

2. **AWS Account**: Create an AWS account and familiarize yourself with AWS EMR.

3. **Java and Hadoop**: Make sure Java and Hadoop are installed and configured correctly. I am using Java11 for this project.

5. **Git and GitHub**: Use Git for version control and host your project repository on GitHub.

6. **IDE**: Use an Integrated Development Environment (IDE) for coding and development.

7. **Spark**: Install Spark on local machine or cluster


## Conclusion

The project shows the importance of using Spark's parellelization capabilites to read data and train models. The successful completion of this project will enhance our understanding of distributed computing and natural language processing while providing practical experience with cloud-based technologies.
