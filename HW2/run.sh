#!/bin/bash

# Build the Docker image
docker build -t hw2_image .

# Run the Docker container
docker run -p 8888:8888 hw2_image

