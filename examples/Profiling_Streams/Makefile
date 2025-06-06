# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: profiling_code_v1 profiling_code_v2 profiling_code_v3

profiling_code_v1.o: profiling_code_v1.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

profiling_code_v1: profiling_code_v1.o
	$(NVXX) profiling_code_v1.o -o profiling_code_v1

profiling_code_v2.o: profiling_code_v2.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

profiling_code_v2: profiling_code_v2.o
	$(NVXX) profiling_code_v2.o -o profiling_code_v2

profiling_code_v3.o: profiling_code_v2.cpp
	$(NVXX) --default-stream per-thread $(NVXXFLAGS) $(NVOPTFLAGS) $< -c -o profiling_code_v3.o

profiling_code_v3: profiling_code_v3.o
	$(NVXX) profiling_code_v3.o -o profiling_code_v3

clean:
	rm -f profiling_code_v1
	rm -f profiling_code_v2
	rm -f profiling_code_v3
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
