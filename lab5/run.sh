#!/bin/bash

# Build the Docker image
docker build -t hw1 .

# Run the Docker container
docker run -p 8888:8888 hw1

