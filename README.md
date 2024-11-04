# Implementing Parallelization using Spark

### Author: Ashish Bhushan
### Email: abhus@uic.edu
### UIN: 654108403

## Introduction

Implementation of Positional Embeddings and training a model based on those embeddings using Spark's parallelization capabilities

Video Link: 
The video explains the deployment of Spark application in the AWS EMR Cluster and the project structure

### Environment
```
OS: Windows 11

IDE: IntelliJ IDEA 2022.2.3 (Ultimate Edition)

SCALA Version: 2.12.18

SBT Version: 1.10.1

Spark Version: 3.5.3

Java Version: 11.0.23
```


### Running the project

1) Clone this repository

```
https://github.com/realAkshaj/CS441-HW2.git
```
2) Open the project in IntelliJ

3) Use Classpath File
   
This is the recommended solution and can be done directly within IntelliJ. Instead of passing a long classpath via the command line, IntelliJ can generate a classpath file.

Steps to Use a Classpath File in IntelliJ:
Open IntelliJ and load your project.

Go to Run > Edit Configurations....

In the Run/Debug Configurations window, select your MainApp run configuration.

Scroll down to the Configuration tab, and find the option Shorten command line.

From the dropdown, choose JAR manifest or classpath file:

Classpath file is the most common and typically recommended option. It allows IntelliJ to store the classpath in a temporary file and reference it rather than passing it directly to the command line.
JAR manifest works by embedding the classpath inside the JAR manifest (this might require additional configuration if you're building a JAR).

Click Apply and then OK.

Re-run your application.   

4) Run MainApp.

5) You can pass arguments to make the code compatible with other frameworks - pass nothing to run on local, "aws" for aws and "spark" for spark



```

## Project Structure

The project comprises the following key components:


- Positional Embedding Generation: Embeddings are generated based on the order of the sentences for contextual purposes and generation of Target Embeddings using Spark's parallelization capabilities

- Training and Storing of Model: A neural network is trained based on the positional embeddings and is stored using Spark's parallelization capabilities

- Generation of  Metrics: the metrics of the model are generated and stored in a file called metrics.csv (some of the metrics are visible in the logs of the Spark runtime)

## Prerequisites

Before starting the project, ensure that you have the necessary tools and accounts set up:

1. **Hadoop**: Set up Hadoop on your local machine or cluster.

2. **AWS Account**: Create an AWS account and familiarize yourself with AWS EMR.

3. **Java and Hadoop**: Make sure Java and Hadoop are installed and configured correctly.

5. **Git and GitHub**: Use Git for version control and host your project repository on GitHub.

6. **IDE**: Use an Integrated Development Environment (IDE) for coding and development.

7. **Spark**: Install Spark on local machine or cluster


## Conclusion

The project shows the importance of using Spark's parellelization capabilites to read data and train models.

For detailed instructions on how to set up and run the project, please refer to the project's documentation and README files.

**Note:** This README provides an overview of the project. For detailed documentation and instructions, refer to the project's YouTube video link and src files
