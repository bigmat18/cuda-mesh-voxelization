# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: recursive_helloworld

recursive.o: recursive.cu
    $(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -dc

recursive_helloworld.o: recursive_helloworld.cpp
    $(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

recursive_helloworld: recursive_helloworld.o recursive.o
    $(NVXX) $(NVOPTFLAGS) recursive_helloworld.o recursive.o -o recursive_helloworld

clean:
    rm -f recursive_helloworld
    rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
