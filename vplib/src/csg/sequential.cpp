#include "proc_utils.h"
#include "profiling.h"
#include <csg/csg.h>

namespace CSG {
    
template <Types type, typename T, typename func>
void Compute(HostVoxelsGrid<T>& grid1, HostVoxelsGrid<T>& grid2, func Op) 
{
    PROFILING_SCOPE("SequentialCSG");

    {
        PROFILING_SCOPE("SequentialCSG::Processing");
        auto& grid1V = grid1.View();
        auto& grid2V = grid2.View();

        const size_t numWord = (grid1V.Size() + grid1V.WordSize() - 1) / grid1V.WordSize();
        for(int i = 0; i < numWord; i++) {

            const int voxelIndex = i * grid1V.WordSize();

            const int z = voxelIndex / (grid1V.VoxelsPerSide() * grid1V.VoxelsPerSide());
            const int y = (voxelIndex % (grid1V.VoxelsPerSide() * grid1V.VoxelsPerSide())) / grid1V.VoxelsPerSide();
            const int x = voxelIndex % grid1V.VoxelsPerSide();

            T word = 0;
            for(int i = 0; i < grid2V.WordSize(); ++i) 
                word |= grid2V.Voxel(x + i, y, z) << i;

            Op(grid1V.Word(x, y, z), word);
        }
    }
}

////////////////////////////// Union OP ///////////////////////////////
template void Compute<Types::SEQUENTIAL, uint32_t, Union<uint32_t>>
(HostVoxelsGrid<uint32_t>& grid1, HostVoxelsGrid<uint32_t>& grid2, Union<uint32_t>);

template void Compute<Types::SEQUENTIAL, uint64_t, Union<uint64_t>>
(HostVoxelsGrid<uint64_t>& grid1, HostVoxelsGrid<uint64_t>& grid2, Union<uint64_t>);
////////////////////////////// Union OP ///////////////////////////////


////////////////////////////// Intersection OP ///////////////////////////////
template void Compute<Types::SEQUENTIAL, uint32_t, Intersection<uint32_t>>
(HostVoxelsGrid<uint32_t>& grid1, HostVoxelsGrid<uint32_t>& grid2, Intersection<uint32_t>);

template void Compute<Types::SEQUENTIAL, uint64_t, Intersection<uint64_t>>
(HostVoxelsGrid<uint64_t>& grid1, HostVoxelsGrid<uint64_t>& grid2, Intersection<uint64_t>);
////////////////////////////// Intersection OP ///////////////////////////////


////////////////////////////// Difference OP ///////////////////////////////
template void Compute<Types::SEQUENTIAL, uint32_t, Difference<uint32_t>>
(HostVoxelsGrid<uint32_t>& grid1, HostVoxelsGrid<uint32_t>& grid2, Difference<uint32_t>);

template void Compute<Types::SEQUENTIAL, uint64_t, Difference<uint64_t>>
(HostVoxelsGrid<uint64_t>& grid1, HostVoxelsGrid<uint64_t>& grid2, Difference<uint64_t>);
////////////////////////////// Difference OP ///////////////////////////////
}
