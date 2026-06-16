#!/bin/bash

IMAGE_NAME="person-detection-app"
CONTAINER_NAME="person-detection-container"

echo "Building Podman image..."
podman build -t $IMAGE_NAME .

echo "Running container with webcam and display access..."
# X11認証を許可
xhost +local:

podman run --rm -it \
    --name $CONTAINER_NAME \
    --device=/dev/video0:/dev/video0 \
    --device=/dev/video1:/dev/video1 \
    --device=/dev/snd:/dev/snd \
    -e DISPLAY=$DISPLAY \
    -e XDG_RUNTIME_DIR=/run/user/$(id -u) \
    -e PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /run/user/$(id -u):/run/user/$(id -u):rw \
    --security-opt label=disable \
    --privileged \
    $IMAGE_NAME

# 終了後にX11認証を元に戻す
xhost -local:
