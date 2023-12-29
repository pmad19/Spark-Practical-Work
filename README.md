# Spark Practical Work
This project has been developed for Big Data subject from MSc in Data Science of 
Universidad Politéncica de Madrid, done during the First Semester 2023/2024

## Introduction
The objective of this work is to help students to put into practice the concepts learnt during the
theory lessons, and to get proficiency in the use of Spark and other related Big Data
technologies. In this exercise the students are
required to develop a Spark application that creates a machine learning model for a real-world
problem, using real-world data: Predicting the arrival delay of commercial flights.

## The Problem
The basic problem of this exercise is to create a model capable of predicting the arrival delay
time of a commercial flight, given a set of parameters known at time of take-off. To do that,
students will use publicly available data from commercial USA domestic flights. The main result
of this work will be a Spark application, programmed to perform the following tasks:
- Load the input data, previously stored at a known location.
- Select, process and transform the input variables, to prepare them for training the model.
- Perform some basic analysis of each input variable.
- Create a machine learning model that predicts the arrival delay time.
- Validate the created model and provide some measure of its accuracy.

## The Data
For this exercise, students will use data published by the US Department of Transportation. This
data can be downloaded from the following URL:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7
The dataset is divided into several independent files, to make download easier. You do not need
to download and use the entire dataset. A small piece should be sufficient, one that fits in your
development environment and does not take too long to process. The Spark application you
develop, however, should be able to work with any subset of this dataset, and not be limited to a
specific piece.

## How to run the project
### Create a venv
    cd Spark-Practical-Work
    python3 -m venv venv
    source venv/bin/activate
    pip install pyspark

### Run the project
##### Load the virtual environment (just once)
    cd Spark-Practical-Work
    source venv/bin/activate
##### Run the application
    cd Spark-Practical-Work/src/main/resources
    sh install.sh && run.sh
    
