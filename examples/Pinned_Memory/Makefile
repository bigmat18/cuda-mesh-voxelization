# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_61 --gpu-code=sm_61 --ptxas-options=-v
CUDA_STATIC_LIB_PATH = $(shell find /nix/store -name libcudart_static.a | head -n1 | xargs dirname)

all: pageable_test pinned_test pinned_test_2

pageable_test.o: pageable_test.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

pageable_test: pageable_test.o
	$(NVXX) pageable_test.o -o pageable_test -L$(CUDA_STATIC_LIB_PATH)

pinned_test.o: pinned_test.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

pinned_test: pinned_test.o
	$(NVXX) pinned_test.o -o pinned_test -L$(CUDA_STATIC_LIB_PATH)

pinned_test_2.o: pinned_test_2.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

pinned_test_2: pinned_test_2.o
	$(NVXX) pinned_test_2.o -o pinned_test_2 -L$(CUDA_STATIC_LIB_PATH)

clean:
	rm -f pageable_test
	rm -f pinned_test
	rm -f pinned_test_2
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
