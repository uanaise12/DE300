#!/bin/bash

# Define the path to the Python script
SCRIPT="s3://de300spring2024/anaise/lab7/spark_ml_pipeline.py"

# Define EMR Cluster ID
CLUSTER_ID="j-2XQ4Z9TJ703PD"

# Define the log path
LOG_URI="s3://de300spring2024/anaise/lab7/logs/"

# Submit the job to the EMR cluster
aws emr add-steps --cluster-id $CLUSTER_ID --steps Type=Spark,Name="Spark ML Job",Action
