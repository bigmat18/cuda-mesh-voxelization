#ifndef JFA_H
#define JFA_H

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector_types.h>

#include <mesh/mesh.h>
#include <grid/voxels_grid.h>
#include <grid/grid.h>
#include <profiling.h>
#include <proc_utils.h>

namespace JFA { 

__host__ __device__ inline float CalculateDistance(Position p0, Position p1) 
{ return ((p1.X - p0.X) * (p1.X - p0.X)) + ((p1.Y - p0.Y) * (p1.Y - p0.Y)) + ((p1.Z - p0.Z) * (p1.Z - p0.Z)); }


constexpr int TILE_DIM_INIT = 3;
template <typename T, int TILE_DIM = TILE_DIM_INIT>
__global__ void InizializationTiled(const VoxelsGrid<T, true> grid, Grid<float> sdf, Grid<Position> positions);

template <typename T>
__global__ void InizializationNaive(const VoxelsGrid<T, true> grid, Grid<float> sdf, Grid<Position> positions);

template <typename T>
__global__ void ProcessingNaive(const int K, const VoxelsGrid<T, true> grid,
                                const Grid<float> inSDF, const Grid<Position> inPositions,
                                Grid<float> outSDF, Grid<Position> outPositions);

template <typename T, int TILE_DIM = 14>
__global__ void ProcessingTiled(const int K, const VoxelsGrid<T, true> grid,
                                const Grid<float> inSDF, const Grid<Position> inPositions,
                                Grid<float> outSDF, Grid<Position> outPositions);

template <Types type, typename T>
void Compute(HostVoxelsGrid<T>& grid, HostGrid<float>& sdf);

};

#endif // !JFA_H
