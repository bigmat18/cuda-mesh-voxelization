# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_61 --gpu-code=sm_61
CUDA_STATIC_LIB_PATH = $(shell find /nix/store -name libcudart_static.a | head -n1 | xargs dirname)

all: printRank cg_reduce_block cg_reduce_warp cg_tile_reduce

printRank.o: printRank.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

printRank: printRank.o
	$(NVXX) printRank.o -o printRank -L$(CUDA_STATIC_LIB_PATH)

cg_reduce_block.o: cg_reduce.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) -DBLOCK_BASIS $< -c -o $@

cg_reduce_block: cg_reduce_block.o
	$(NVXX) cg_reduce_block.o -o cg_reduce_block -L$(CUDA_STATIC_LIB_PATH)

cg_reduce_warp.o: cg_reduce.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) -DWARP_BASIS $< -c -o $@

cg_reduce_warp: cg_reduce_warp.o
	$(NVXX) cg_reduce_warp.o -o cg_reduce_warp -L$(CUDA_STATIC_LIB_PATH)

cg_tile_reduce.o: cg_tile_reduce.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

cg_tile_reduce: cg_tile_reduce.o
	$(NVXX) cg_tile_reduce.o -o cg_tile_reduce -L$(CUDA_STATIC_LIB_PATH)

clean:
	rm -f printRank
	rm -f cg_reduce_block
	rm -f cg_reduce_warp
	rm -f cg_tile_reduce
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
