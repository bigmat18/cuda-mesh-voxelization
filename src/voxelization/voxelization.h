#ifndef VOXELIZATION_H
#define VOXELIZATION_H

#include <voxels_grid.h>
#include <mesh/mesh.h>


class Voxelization {
    void SequentialSolidVoxelization();

    void NaiveSolidVoxelization();

    void TiledSolidVoxelization();

public:
    enum class Types {
        SEQUENTIAL, NAIVE, TAILED
    };

    template<Types type, typename T>
    static bool Compute(VoxelsGrid<T, false> grid, const Mesh& mesh) 
    requires (type == Types::SEQUENTIAL);


    template<Types type, typename T>
    static bool Compute(VoxelsGrid<T, true> grid, const Mesh& mesh, bool device, int blockSize) 
    requires (type == Types::NAIVE);


    template<Types type, typename T>
    static bool Compute(VoxelsGrid<T, true> grid, const Mesh& mesh, bool device, int blockSize) 
    requires (type == Types::TAILED);
}


#endif // !VOXELIZATION_H,
