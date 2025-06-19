#ifndef SOLID_VOXEL_H
#define SOLID_VOXEL_H

enum VoxelizationTypes {
    SEQUENTIAL, NAIVE, TAILED
};

void SequentialSolidVoxelization()
{

}

void NaiveSolidVoxelization() 
{

}


template<VoxelizationTypes type>
bool SolidVoxelization() 
{
    return true;
}

#endif // !SOLID_VOXEL_H
