#ifndef CSG_H
#define CSG_H

#include <grid/voxels_grid.h>
#include <proc_utils.h>
#include <debug_utils.h>

namespace CSG {

enum class Op {
    VOID, UNION, INTERSECTION, DIFFERENCE
};

template <typename T> 
struct Union {
    __host__ __device__
    void operator() (T& el, T value) { el |= value; }
};

template <typename T> 
struct Intersection {
    __host__ __device__
    void operator() (T& el, T value) { el &= value; }
};

template <typename T> 
struct Difference {
    __host__ __device__
    void operator() (T& el, T value) { el &= ~value; }
};
    
template <typename T, typename func>
__global__ void ProcessingNaive(VoxelsGrid<T, true> grid1, VoxelsGrid<T, true> grid2, func Op);

template <Types type, typename T, typename func>
void Compute(HostVoxelsGrid<T>& grid1, HostVoxelsGrid<T>& grid2, func Op);

}

#endif // !CSG_H

