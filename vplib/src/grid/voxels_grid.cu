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
    const VoxelsGrid v = device.View();
    const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(v.mSizeX, v.mSizeY, v.mSizeZ);

    mData = std::make_unique<T[]>(storageSize);
    mView = VoxelsGrid<T>(mData.get(), v.mSizeX, v.mVoxelSize); 
    mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ());
    gpuAssert(cudaMemcpy(mData.get(), device.mData, storageSize * sizeof(T), cudaMemcpyDeviceToHost));
}

template <VGType T>
HostVoxelsGrid<T>::HostVoxelsGrid(const HostVoxelsGrid<T>& other) 
{
    const VoxelsGrid v = other.View();
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
    const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(
        voxelsPerSideX, voxelsPerSideY, voxelsPerSideZ) * sizeof(T);

    gpuAssert(cudaMalloc((void**) &mData, storageSize));   
    mView = VoxelsGrid<T, true>(mData, voxelsPerSideX, voxelsPerSideY, voxelsPerSideZ, voxelSize);
    gpuAssert(cudaMemset(mData, 0, storageSize));
}

template <VGType T>
DeviceVoxelsGrid<T>::DeviceVoxelsGrid(const size_t voxelsPerSide, const float voxelSize)
{
    const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(voxelsPerSide) * sizeof(T);

    gpuAssert(cudaMalloc((void**) &mData, storageSize));   
    mView = VoxelsGrid<T, true>(mData, voxelsPerSide, voxelSize);
    gpuAssert(cudaMemset(mData, 0, storageSize));
}

template <VGType T>
DeviceVoxelsGrid<T>::DeviceVoxelsGrid(const HostVoxelsGrid<T>& host) 
{
    const VoxelsGrid v = host.View();
    const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(
        v.mSizeX, v.mSizeY, v.mSizeZ) * sizeof(T);

    gpuAssert(cudaMalloc((void**) &mData, storageSize));
    mView = VoxelsGrid<T, true>(mData, v.mSizeX, v.mSizeY, v.mSizeZ, v.mVoxelSize);
    mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ());
    gpuAssert(cudaMemcpy(mData, host.mData.get(), storageSize, cudaMemcpyHostToDevice));
}

template <VGType T>
DeviceVoxelsGrid<T>::DeviceVoxelsGrid(const DeviceVoxelsGrid<T>& other)
{
    const VoxelsGrid v = other.View();
    const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(
            v.mSizeX, v.mSizeY, v.mSizeZ) * sizeof(T);

    gpuAssert(cudaMalloc((void**) &mData, storageSize));
    mView = VoxelsGrid<T, true>(mData, v.mSizeX, v.mSizeY, v.mSizeZ, v.mVoxelSize);
    mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ());
    gpuAssert(cudaMemcpy(mData, other.mData, storageSize, cudaMemcpyDeviceToDevice));
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

