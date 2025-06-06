# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: naive_transpose smem_transpose

naive_transpose.o: naive_transpose.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

naive_transpose: naive_transpose.o
	$(NVXX) naive_transpose.o -o naive_transpose

smem_transpose.o: smem_transpose.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

smem_transpose: smem_transpose.o
	$(NVXX) smem_transpose.o -o smem_transpose

clean:
	rm -f naive_transpose
	rm -f smem_transpose
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
