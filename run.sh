#!/bin/bash

IMAGE_NAME="person-detection-app"
CONTAINER_NAME="person-detection-container"

echo "Building Podman image..."
podman build -t $IMAGE_NAME .

echo "Running container with webcam and display access..."
podman run --rm -it \
    --name $CONTAINER_NAME \
    --device=/dev/video0:/dev/video0 \
    --device=/dev/video1:/dev/video1 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --security-opt label=disable \
    --privileged \
    $IMAGE_NAME
