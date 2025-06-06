# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: naive_reduction blocked_reduction strided_reduction smem_reduction coarse_reduction

naive_reduction.o: naive_reduction.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

naive_reduction: naive_reduction.o
	$(NVXX) naive_reduction.o -o naive_reduction

blocked_reduction.o: blocked_reduction.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

blocked_reduction: blocked_reduction.o
	$(NVXX) blocked_reduction.o -o blocked_reduction

strided_reduction.o: strided_reduction.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

strided_reduction: strided_reduction.o
	$(NVXX) strided_reduction.o -o strided_reduction

smem_reduction.o: smem_reduction.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

smem_reduction: smem_reduction.o
	$(NVXX) smem_reduction.o -o smem_reduction

coarse_reduction.o: coarse_reduction.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

coarse_reduction: coarse_reduction.o
	$(NVXX) coarse_reduction.o -o coarse_reduction

clean:
	rm -f naive_reduction
	rm -f blocked_reduction
	rm -f strided_reduction
	rm -f smem_reduction
	rm -f coarse_reduction
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
