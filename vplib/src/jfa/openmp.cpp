#include "mesh/mesh.h"
#include "profiling.h"
#include <jfa/jfa.h>
#include <omp.h>
    
namespace JFA {

template <Types type, typename T>
void Compute(HostVoxelsGrid<T>& grid, HostGrid<float>& sdf)
{
    PROFILING_SCOPE("OpenmpJFA");
    auto& gridV = grid.View();

    HostGrid<Position> positions;
    {
        PROFILING_SCOPE("OpenmpJFA::Memory");
        positions = HostGrid<Position>(grid.View().VoxelsPerSide());
    }

    {
        PROFILING_SCOPE("OpenmpJFA::Initialization");
        auto& sdfV = sdf.View();
        auto& positionsV = positions.View();

        #pragma omp parallel for collapse(1) schedule(static)
        for (int voxelZ = 0; voxelZ < gridV.SizeZ(); ++voxelZ) {
            for (int voxelY = 0; voxelY < gridV.SizeY(); ++voxelY) {
                for (int voxelX = 0; voxelX < gridV.SizeX(); ++voxelX) {

                    if(!gridV.Voxel(voxelX, voxelY, voxelZ))
                        continue;
                                
                    bool found = false;
                    Position pos = Position(gridV.OriginX() + (voxelX * gridV.VoxelSize()),
                                            gridV.OriginY() + (voxelY * gridV.VoxelSize()),
                                            gridV.OriginZ() + (voxelZ * gridV.VoxelSize()));

                    for(int z = -1; z <= 1; z++) {
                        for(int y = -1; y <= 1; y++) {
                            for(int x = -1; x <= 1; x++) {
                                if(x == 0 && y == 0 && z == 0)
                                    continue;

                                int nx = voxelX + x;
                                int ny = voxelY + y;
                                int nz = voxelZ + z;

                                bool isBorder = nx < 0 || nx >= gridV.VoxelsPerSide() || 
                                                ny < 0 || ny >= gridV.VoxelsPerSide() || 
                                                nz < 0 || nz >= gridV.VoxelsPerSide();

                                if(isBorder || !gridV.Voxel(nx, ny, nz))
                                    found = true;
                            }
                        }
                    }
                    if(found) {
                        sdfV(voxelX, voxelY, voxelZ) = 0.0f;
                        positionsV(voxelX, voxelY, voxelZ) = pos;
                    } else {
                        sdfV(voxelX, voxelY, voxelZ) = INFINITY;
                    }
                }
            }
        }
    }
    

    {
        PROFILING_SCOPE("Openmp::Processing");
        HostGrid sdfApp(sdf);
        HostGrid positionsApp(positions);

        for(int k = gridV.VoxelsPerSide() / 2; k >= 1; k/=2) { 

            #pragma omp parallel for collapse(3) schedule(static)
            for (int voxelZ = 0; voxelZ < gridV.SizeZ(); ++voxelZ)  {
                for (int voxelY = 0; voxelY < gridV.SizeY(); ++voxelY) {
                    for (int voxelX = 0; voxelX < gridV.SizeX(); ++voxelX) {


                        Position voxelPos = Position(gridV.OriginX() + (voxelX * gridV.VoxelSize()),
                                                     gridV.OriginY() + (voxelY * gridV.VoxelSize()),
                                                     gridV.OriginZ() + (voxelZ * gridV.VoxelSize()));
 
                        bool findNewBest = false;  
                        float bestDistance = sdf.View()(voxelX, voxelY, voxelZ);
                        Position bestPosition;
                        for(int z = -1; z <= 1; z++) {
                            for(int y = -1; y <= 1; y++) {
                                for(int x = -1; x <= 1; x++) {
                                    if(x == 0 && y == 0 && z == 0)
                                        continue;

                                    int nx = voxelX + (x * k);
                                    int ny = voxelY + (y * k);
                                    int nz = voxelZ + (z * k);

                                    if(nx < 0 || nx >= gridV.VoxelsPerSide() ||
                                        ny < 0 || ny >= gridV.VoxelsPerSide() ||
                                        nz < 0 || nz >= gridV.VoxelsPerSide())
                                        continue; 
  
                                    float seed = sdf.View()(nx, ny, nz);
                                    if(fabs(seed) < INFINITY) {
                                        Position seedPos = positions.View()(nx, ny, nz);

                                        float distance = CalculateDistance(voxelPos, seedPos);
                                        if(distance < fabs(bestDistance)) {
                                            findNewBest = true;
                                            bestDistance = copysignf(distance, bestDistance);
                                            bestPosition = seedPos;
                                      }
                                    }
                                }
                            }
                        }

                        if (findNewBest) {
                            sdfApp.View()(voxelX, voxelY, voxelZ) = bestDistance;
                            positionsApp.View()(voxelX, voxelY, voxelZ) = bestPosition;
                        }
                    }
                }
            }
            sdf = sdfApp;
            positions = positionsApp;
        }
    }
}


template void Compute<Types::OPENMP, uint32_t>
(HostVoxelsGrid<uint32_t>&, HostGrid<float>&);

template void Compute<Types::OPENMP, uint64_t>
(HostVoxelsGrid<uint64_t>&, HostGrid<float>&);
}
