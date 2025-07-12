#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector_types.h>

#include <mesh/mesh.h>
#include <grid/voxels_grid.h>
#include <profiling.h>
#include <proc_utils.h>

#ifndef JFA_H
#define JFA_H

namespace JFA {

struct SDF {
    int x = -1;
    int y = -1;
    int z = -1;
    float distance = std::numeric_limits<float>::infinity();
};

__host__ __device__ inline float CalculateDistance(Position p0, Position p1) 
{ return std::sqrt(std::pow(p1.X - p0.X, 2) + std::pow(p1.Y - p0.Y, 2) + std::pow(p1.Z - p0.Z, 2)); }


template <typename T, int TILE_DIM = 4>
__global__ void InizializationTiled(const VoxelsGrid<T, true> grid, SDF* SDFValues);

template <typename T>
__global__ void InizializationNaive(const VoxelsGrid<T, true> grid, SDF* SDFValues);

template <typename T>
__global__ void ProcessingNaive(const int K, const VoxelsGrid<T, true> grid, const SDF* valuesIn, SDF* valuesOut);


template <Types type, typename T>
void Compute(DeviceVoxelsGrid<T>& grid, std::vector<SDF>& sdfValues);

};

#endif // !JFA_H
