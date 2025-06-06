# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: image_blur_gpu image_blur_cpu

image_blur_gpu.o: image_blur_gpu.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

image_blur_gpu: image_blur_gpu.o
	$(NVXX) image_blur_gpu.o -o image_blur_gpu

image_blur_cpu.o: image_blur_cpu.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

image_blur_cpu: image_blur_cpu.o
	$(NVXX) image_blur_cpu.o -o image_blur_cpu

clean:
	rm -f image_blur_gpu
	rm -f image_blur_cpu
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
