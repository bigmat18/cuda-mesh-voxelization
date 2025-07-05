#!/bin/bash
xhost +
docker run -it --rm \
       --device nvidia.com/gpu=all \
       -e DISPLAY=$DISPLAY \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       --cap-add=SYS_ADMIN \
       --security-opt \
       seccomp=unconfined \
       -v "$(pwd)":/mnt \
       --network=host \
       nsight
