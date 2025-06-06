# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O0 --gpu-architecture=compute_61 --gpu-code=sm_61 --ptxas-options=-v
CUDA_STATIC_LIB_PATH = $(shell find /nix/store -name libcudart_static.a | head -n1 | xargs dirname)

all: occupancy

occupancy.o: occupancy.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

occupancy: occupancy.o
	$(NVXX) occupancy.o -o occupancy -L$(CUDA_STATIC_LIB_PATH)

clean:
	rm -f occupancy
	rm *.o

.DEFAULT_GOAL := all
.PHONY: all clean
