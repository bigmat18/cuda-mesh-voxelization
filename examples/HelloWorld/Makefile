# Author: Gabriele Mencagli
NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O0 --gpu-architecture=compute_61 --gpu-code=sm_61 --ptxas-options=-v
CUDA_STATIC_LIB_PATH = $(shell find /nix/store -name libcudart_static.a | head -n1 | xargs dirname)

all: hello_world

hello_world.o: hello_world.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

hello_world: hello_world.o
	$(NVXX) hello_world.o -o hello_world -L$(CUDA_STATIC_LIB_PATH)

clean:
	rm -f hello_world
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
