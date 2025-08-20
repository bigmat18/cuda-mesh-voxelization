#!/bin/bash
xhost +
docker run -it \
       --device nvidia.com/gpu=all \
       --device /dev/net/tun \
       -e DISPLAY=$DISPLAY \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       --cap-add=SYS_ADMIN \
       --cap-add=NET_ADMIN \
       --security-opt \
       seccomp=unconfined \
       -v /home/bigmat18:/home/bigmat18 \
       --network=host \
        --dns=8.8.8.8 --dns=1.1.1.1 \
       nsight-tunnel

