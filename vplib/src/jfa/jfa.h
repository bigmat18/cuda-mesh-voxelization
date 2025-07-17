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
{ return std::sqrt(std::pow(p1.X - p0.X, 2) + std::pow(p1.Y - p0.Y, 2) + std::pow(p1.Z - p0.Z, 2)); }


template <typename T, int TILE_DIM = 3>
__global__ void InizializationTiled(const VoxelsGrid<T, true> grid, Grid<float> sdf, Grid<Position> positions);

template <typename T>
__global__ void InizializationNaive(const VoxelsGrid<T, true> grid, Grid<float> sdf, Grid<Position> positions);

template <typename T>
__global__ void ProcessingNaive(const VoxelsGrid<T, true> grid,
                                Grid<float> sdf, Grid<Position> positions);

template <typename T>
__global__ void ProcessingTiled(const int K, const int inTileSize, const VoxelsGrid<T, true> grid,
                                const Grid<float> inSDF, const Grid<Position> inPositions,
                                Grid<float> outSDF, Grid<Position> outPositions);

template <Types type, typename T>
void Compute(DeviceVoxelsGrid<T>& grid, DeviceGrid<float>& sdf, DeviceGrid<Position>& positions);

template <Types type, typename T>
void Compute(HostVoxelsGrid<T>& grid, HostGrid<float>& sdf, HostGrid<Position>& positions);

};

#endif // !JFA_H
