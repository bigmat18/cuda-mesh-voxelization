# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_61 --gpu-code=sm_61
CUDA_STATIC_LIB_PATH = $(shell find /nix/store -name libcudart_static.a | head -n1 | xargs dirname)

all: image_flip_gpu

image_flip_gpu.o: image_flip_gpu.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

image_flip_gpu: image_flip_gpu.o
	$(NVXX) image_flip_gpu.o -o image_flip_gpu -L$(CUDA_STATIC_LIB_PATH)

# image_flip_cpu.o: image_flip_cpu.cpp
# 	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

# image_flip_cpu: image_flip_cpu.o
# 	$(NVXX) image_flip_cpu.o -o image_flip_cpu

clean:
	rm -f image_flip_gpu
	rm -f image_flip_cpu
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
