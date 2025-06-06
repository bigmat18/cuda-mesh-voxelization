# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: naive_convolution tiled_convolution const_tiled_convolution

naive_convolution.o: naive_convolution.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

naive_convolution: naive_convolution.o
	$(NVXX) naive_convolution.o -o naive_convolution

tiled_convolution.o: tiled_convolution.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

tiled_convolution: tiled_convolution.o
	$(NVXX) tiled_convolution.o -o tiled_convolution

const_tiled_convolution.o: const_tiled_convolution.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

const_tiled_convolution: const_tiled_convolution.o
	$(NVXX) const_tiled_convolution.o -o const_tiled_convolution

clean:
	rm -f naive_convolution
	rm -f tiled_convolution
	rm -f const_tiled_convolution
	rm *.o

.DEFAULT_GOAL := all
.PHONY: all clean
