#!/bin/bash
DOCKER_NAME="mask2former_container"
echo "Executing the following command - docker build -t $DOCKER_NAME"
docker build -t $DOCKER_NAME .
