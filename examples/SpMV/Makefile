# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGS      = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80

all: coo_spmv csr_spmv ell_spmv

coo_spmv.o: coo_spmv.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

coo_spmv: coo_spmv.o
	$(NVXX) coo_spmv.o -o coo_spmv

csr_spmv.o: csr_spmv.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

csr_spmv: csr_spmv.o
	$(NVXX) csr_spmv.o -o csr_spmv

ell_spmv.o: ell_spmv.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGS) $< -c

ell_spmv: ell_spmv.o
	$(NVXX) ell_spmv.o -o ell_spmv

clean:
	rm -f coo_spmv
	rm -f csr_spmv
	rm -f ell_spmv
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
