# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: naive_stencils tiled_stencils coarsed_tiled_stencils coarsed_r_tiled_stencils

naive_stencils.o: naive_stencils.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

naive_stencils: naive_stencils.o
	$(NVXX) naive_stencils.o -o naive_stencils

tiled_stencils.o: tiled_stencils.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

tiled_stencils: tiled_stencils.o
	$(NVXX) tiled_stencils.o -o tiled_stencils

coarsed_tiled_stencils.o: coarsed_tiled_stencils.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

coarsed_tiled_stencils: coarsed_tiled_stencils.o
	$(NVXX) coarsed_tiled_stencils.o -o coarsed_tiled_stencils

coarsed_r_tiled_stencils.o: coarsed_r_tiled_stencils.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

coarsed_r_tiled_stencils: coarsed_r_tiled_stencils.o
	$(NVXX) coarsed_r_tiled_stencils.o -o coarsed_r_tiled_stencils

clean:
	rm -f naive_stencils
	rm -f tiled_stencils
	rm -f coarsed_tiled_stencils
	rm -f coarsed_r_tiled_stencils
	rm *.o

.DEFAULT_GOAL := all
.PHONY: all clean
