#ifndef GRID_H
#define GRID_H

#include <debug_utils.h>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <span>
#include <sys/types.h>

template <typename T>
class Grid 
{
    using uint = unsigned int;

    size_t mSizeX = 1;
    size_t mSizeY = 1;
    size_t mSizeZ = 1;
    std::span<T> mGrid;
    
public:

    __host__ __device__
    Grid() = default;

    __host__ __device__
    Grid(T* data, const size_t sizeX, const size_t sizeY, const size_t sizeZ) :
        mSizeX(sizeX), mSizeY(sizeY), mSizeZ(sizeZ) 
    {
        mGrid = std::span<T>(data, Size());
    }

    __host__ __device__
    T operator()(const uint x, const uint y, const uint z) const
    {
        assert(x < mSizeX); assert(y < mSizeY); assert(z < mSizeZ);
        return mGrid[Index(x, y, z)];
    }  

    __host__ __device__
    T& operator()(const uint x, const uint y, const uint z)
    {
        assert(x < mSizeX); assert(y < mSizeY); assert(z < mSizeZ);
        return mGrid[Index(x, y, z)];
    }  

    __host__ __device__
    inline size_t Size() const { return mSizeX * mSizeY * mSizeZ; }

    __host__ __device__
    inline size_t SizeX() const { return mSizeX; }

    __host__ __device__
    inline size_t SizeY() const { return mSizeY; }

    __host__ __device__
    inline size_t SizeZ() const { return mSizeZ; }

    __host__ __device__
    inline void Print() const 
    {
        for(int z = 0; z < mSizeZ; ++z) {
            for (int y = 0; y < mSizeY; ++y) {
                for (int x = 0; x < mSizeX; ++x) {
                    printf("%d ", (*this)(x, y, z));    
                }
                printf("\n");
            }
            printf("\n");
        }
    }

private:

    __host__ __device__
    inline uint Index(const uint x, const uint y, const uint z) const 
    {
        return x + (y * mSizeX) + (z * mSizeX * mSizeY);
    }
};



template <typename T>
class HostGrid;

template <typename T>
class DeviceGrid;


template <typename T>
class HostGrid 
{
    std::unique_ptr<T[]> mData;
    Grid<T> mView;

public:
    HostGrid(const size_t sizeX, const size_t sizeY, const size_t sizeZ)
    {
        mData = std::make_unique<T[]>(sizeX * sizeY * sizeZ);
        mView = Grid(mData.get(), sizeX, sizeY, sizeZ);
        std::fill(mView.mGrid.begin(), mView.mGrid.end(), 0);
    }

    HostGrid(const DeviceGrid<T>& device) 
    {
        mData = std::make_unique<T[]>(device.mSizeX * device.mSizeY * device.mSizeZ);
        mView = Grid<T>(mData.get(), device.mSizeX, device.mSizeY, device.mSizeZ);
        gpuAssert(cudaMemcpy(mData.get(), device.mData, mView.Size() * sizeof(T), cudaMemcpyDeviceToHost));
    }

    HostGrid(const HostGrid& other) 
    {
        mData = std::make_unique<T[]>(other.mSizeX * other.mSizeY * other.mSizeZ);
        mView = Grid(mData.get(), other.mSizeX, other.mSizeY, other.mSizeZ);
        std::copy(mData.get(), mData.get() + mView.Size(), other.mData.get()); 
    }

    HostGrid(HostGrid&& other) { swap(other); }

    HostGrid& operator=(const HostGrid& other) { swap(other); return *this; }

    void swap(HostGrid& other)
    {
        using std::swap;
        swap(mData, other.mData);
        swap(mView, other.mView);
    }

    friend void swap(HostGrid& first, HostGrid& second) { first.swap(second); }

    inline Grid<T>& View() { return mView; }

    inline const Grid<T>& View() const { return mView; }

    friend class DeviceGrid<T>;
};


template <typename T>
class DeviceGrid 
{
    T* mData = nullptr;
    Grid<T> mView;

public:
    DeviceGrid(const size_t sizeX, const size_t sizeY, const size_t sizeZ)
    {
        const size_t storageSize = (sizeX * sizeY * sizeZ) * sizeof(T);

        gpuAssert(cudaMalloc((void**) &mData, storageSize));   
        gpuAssert(cudaMemset(mData, 0, storageSize));
        mView = Grid(mData, sizeX, sizeY, sizeZ);
    }

    DeviceGrid(const HostGrid<T>& host) 
    {
        const size_t storageSize = (host.mSizeX * host.mSizeY * host.mSizeZ) * sizeof(T);

        gpuAssert(cudaMalloc((void**) &mData, storageSize));
        gpuAssert(cudaMemcpy(mData, host.mData.get(), storageSize, cudaMemcpyHostToDevice));
        mView = Grid(mData, host.mSizeX, host.mSizeY, host.mSizeZ);
    }

    DeviceGrid(const DeviceGrid& other) 
    {
        const size_t storageSize = (other.mSizeX * other.mSizeY * other.mSizeZ) * sizeof(T);

        gpuAssert(cudaMalloc((void**) &mData, storageSize));   
        gpuAssert(cudaMemcpy(mData, other.mData, storageSize, cudaMemcpyDeviceToDevice));
        mView = Grid(mData, other.mSizeX, other.mSizeY, other.mSizeZ);
    }

    DeviceGrid(DeviceGrid&& other) { swap(other); }

    DeviceGrid& operator=(const DeviceGrid& other) { swap(other); return *this; }

    ~DeviceGrid()
    {
        if(mData)   
            gpuAssert(cudaFree(mData));
    }   

    void swap(DeviceGrid& other)
    {
        using std::swap;
        swap(mData, other.mData);
        swap(mView, other.mView);
    }

    inline Grid<T>& View() { return mView; }

    inline const Grid<T>& View() const { return mView; }

    friend class HostGrid<T>;
};

#endif // !GRID_H
