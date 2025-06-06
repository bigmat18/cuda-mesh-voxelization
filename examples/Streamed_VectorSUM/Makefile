# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: vectorSUM_naive vectorSUM_streamed

vectorSUM_naive.o: vectorSUM_naive.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

vectorSUM_naive: vectorSUM_naive.o
	$(NVXX) vectorSUM_naive.o -o vectorSUM_naive

vectorSUM_streamed.o: vectorSUM_streamed.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

vectorSUM_streamed: vectorSUM_streamed.o
	$(NVXX) vectorSUM_streamed.o -o vectorSUM_streamed

clean:
	rm -f vectorSUM_naive
	rm -f vectorSUM_streamed
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
