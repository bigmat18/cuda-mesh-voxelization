# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: divergence no_divergence

divergence.o: divergence.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

divergence: divergence.o
	$(NVXX) divergence.o -o divergence

no_divergence.o: no_divergence.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

no_divergence: no_divergence.o
	$(NVXX) no_divergence.o -o no_divergence

clean:
	rm -f divergence
	rm -f no_divergence
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
