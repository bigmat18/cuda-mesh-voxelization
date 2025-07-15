#include <cstdint>
#include <grid/voxels_grid.h>

/////////////////////////////////////////////////////////
//////////////////// HostVoxelsGrid /////////////////////
/////////////////////////////////////////////////////////
template <VGType T>
HostVoxelsGrid<T>::HostVoxelsGrid(const size_t voxelsPerSideX, 
                                  const size_t voxelsPerSideY, 
                                  const size_t voxelsPerSideZ,
                                  const float voxelSize)
{
    mData = std::make_unique<T[]>(VoxelsGrid<T>::CalculateStorageSize(voxelsPerSideX, voxelsPerSideY, voxelsPerSideZ));
    mView = VoxelsGrid<T, false>(mData.get(), voxelsPerSideX, voxelsPerSideY, voxelsPerSideZ, voxelSize);
    std::fill(mView.mGrid.begin(), mView.mGrid.end(), 0);
}

template <VGType T>
HostVoxelsGrid<T>::HostVoxelsGrid(const size_t voxelsPerSide, const float voxelSize)
{
    mData = std::make_unique<T[]>(VoxelsGrid<T>::CalculateStorageSize(voxelsPerSide));
    mView = VoxelsGrid<T, false>(mData.get(), voxelsPerSide, voxelSize);
    std::fill(mView.mGrid.begin(), mView.mGrid.end(), 0);
}

template <VGType T>
HostVoxelsGrid<T>::HostVoxelsGrid(const DeviceVoxelsGrid<T>& device) 
{
    const auto& v = device.View();
    const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(v.mSizeX, v.mSizeY, v.mSizeZ);

    mData = std::make_unique<T[]>(storageSize);
    device.mData.CopyToHost(mData.get(), storageSize);
    mView = VoxelsGrid<T>(mData.get(), v.mSizeX, v.mVoxelSize); 
    mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ());
}

template <VGType T>
HostVoxelsGrid<T>::HostVoxelsGrid(const HostVoxelsGrid<T>& other) 
{
    const auto& v = other.View();
    const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(v.mSizeX, v.mSizeY, v.mSizeZ);

    mData = std::make_unique<T[]>(storageSize);
    mView = VoxelsGrid<T>(mData.get(), v.mSizeX, v.mVoxelSize);
    mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ());
    std::copy(mData.get(), mData.get() + mView.Size(), other.mData.get());
}

template <VGType T>
void HostVoxelsGrid<T>::swap(HostVoxelsGrid<T>& other)
{
    using std::swap;
    swap(mData, other.mData);
    swap(mView, other.mView);
}

template class HostVoxelsGrid<uint32_t>;
template class HostVoxelsGrid<uint64_t>;

/////////////////////////////////////////////////////////
//////////////////// DeviceVoxelsGrid ///////////////////
/////////////////////////////////////////////////////////
template <VGType T>
DeviceVoxelsGrid<T>::DeviceVoxelsGrid(const size_t voxelsPerSideX, 
                                      const size_t voxelsPerSideY, 
                                      const size_t voxelsPerSideZ,
                                      const float voxelSize)
{
    const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(voxelsPerSideX, voxelsPerSideY, voxelsPerSideZ);

    mData = CudaPtr<T>(storageSize); mData.SetMemoryToZero();
    mView = VoxelsGrid<T, true>(mData.get(), voxelsPerSideX, voxelsPerSideY, voxelsPerSideZ, voxelSize);
}

template <VGType T>
DeviceVoxelsGrid<T>::DeviceVoxelsGrid(const size_t voxelsPerSide, const float voxelSize)
{
    const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(voxelsPerSide);

    mData = CudaPtr<T>(storageSize); mData.SetMemoryToZero();
    mView = VoxelsGrid<T, true>(mData.get(), voxelsPerSide, voxelSize);
}

template <VGType T>
DeviceVoxelsGrid<T>::DeviceVoxelsGrid(const HostVoxelsGrid<T>& host) 
{
    const auto& v = host.View();
    const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(v.mSizeX, v.mSizeY, v.mSizeZ);

    mData = CudaPtr<T>(host.mData, storageSize);
    mView = VoxelsGrid<T, true>(mData.get(), v.mSizeX, v.mSizeY, v.mSizeZ, v.mVoxelSize);
    mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ());
}

template <VGType T>
DeviceVoxelsGrid<T>::DeviceVoxelsGrid(const DeviceVoxelsGrid<T>& other)
{
    const auto& v = other.View();
    const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(v.mSizeX, v.mSizeY, v.mSizeZ);

    mData = CudaPtr<T>(other.mData);
    mView = VoxelsGrid<T, true>(mData.get(), v.mSizeX, v.mSizeY, v.mSizeZ, v.mVoxelSize);
    mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ());
}


template <VGType T>
DeviceVoxelsGrid<T>& DeviceVoxelsGrid<T>::operator=(const DeviceVoxelsGrid<T>& other) 
{
    if(this == &other) return *this;
    const auto& v = other.View();
    
    mData = other.mData;
    mView = VoxelsGrid<T, true>(mData.get(), v.mSizeX, v.mSizeY, v.mSizeZ, v.mVoxelSize);
    mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ()); 
    return *this;
}

template <VGType T>
void DeviceVoxelsGrid<T>::swap(DeviceVoxelsGrid<T>& other)
{
    using std::swap;
    swap(mData, other.mData);
    swap(mView, other.mView);
}

template class DeviceVoxelsGrid<uint32_t>;
template class DeviceVoxelsGrid<uint64_t>;

