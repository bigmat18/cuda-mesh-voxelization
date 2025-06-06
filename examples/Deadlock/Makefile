# Author: Gabriele Mencagli

NVXX            = nvcc
NVXXFLAGS       = -std=c++17 -x cu
NVOPTFLAGSNEW   = -w -O3 --gpu-architecture=compute_80 --gpu-code=sm_80
NVOPTFLAGSOLD   = -w -O3 --gpu-architecture=compute_60 --gpu-code=sm_80

all: deadlock_old deadlock_new

deadlock_old.o: deadlock.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGSOLD) $< -c -o deadlock_old.o

deadlock_old: deadlock_old.o
	$(NVXX) deadlock_old.o -o deadlock_old

deadlock_new.o: deadlock.cpp
	$(NVXX) $(NVXXFLAGS) $(NVOPTFLAGSNEW) $< -c -o deadlock_new.o

deadlock_new: deadlock_new.o
	$(NVXX) deadlock_new.o -o deadlock_new

clean:
	rm -f deadlock_old
	rm -f deadlock_new
	rm -f *.o

.DEFAULT_GOAL := all
.PHONY: all clean
