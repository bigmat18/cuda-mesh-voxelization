# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: global_variables

global_variables.o: global_variables.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

global_variables: global_variables.o
	$(NVXX) global_variables.o -o global_variables

clean:
	rm -f global_variables
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
