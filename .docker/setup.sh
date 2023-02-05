#!/bin/bash
# You need to change followings:
DIR_DOCKER_FILE="./.docker"
DOCKER_IMAGE_NAME="pv_segmentation:latest"
DOCKER_CONTAINER_NAME="pv_segmentation"
MEMORY="16G"

echo "*******************************"
echo "Start creating docker image"
docker build --network host -t $DOCKER_IMAGE_NAME $DIR_DOCKER_FILE --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
echo "Finished creating docker image"
echo "*******************************"

# echo "Start creating docker container with GPU"
# docker run --gpus all \
#            --net host \
#            -v $(pwd):/workspace \
#            --shm-size=$MEMORY \
#            --name $DOCKER_CONTAINER_NAME \
#            -itd $DOCKER_IMAGE_NAME
# echo "Finished creating docker container"

echo "Start creating docker container without GPU"
docker run --net host \
           -v $(pwd):/workspace \
           --shm-size=$MEMORY \
           --name $DOCKER_CONTAINER_NAME \
           -itd $DOCKER_IMAGE_NAME
echo "Finished creating docker container"
echo "*******************************"