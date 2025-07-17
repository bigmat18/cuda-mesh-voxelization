#ifndef GRID_H
#define GRID_H

#include <debug_utils.h>
#include <cuda_ptr.h>
#include <mesh/mesh.h>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <span>
#include <sys/types.h>

template <typename T>
class HostGrid;

template <typename T>
class DeviceGrid;

template <typename T>
class Grid 
{
protected:

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
    Grid(T* data, const size_t size) : Grid(data, size, size, size) {}

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
                    PrintValue((*this)(x, y, z));    
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    __host__ __device__
    inline uint Index(const uint x, const uint y, const uint z) const 
    {
        return x + (y * mSizeX) + (z * mSizeX * mSizeY);
    }

    friend class HostGrid<T>;
    friend class DeviceGrid<T>;

public:
    __host__ __device__
    inline void PrintValue(T Value) const { printf("error"); };
};

template <> __host__ __device__
inline void Grid<float>::PrintValue(float value) const { printf("%.2f ", value); } 

template <> __host__ __device__
inline void Grid<int>::PrintValue(int value) const { printf("%d ", value); } 

template <> __host__ __device__
inline void Grid<Position>::PrintValue(Position value) const { printf("(%.2f, %.2f, %.2f) ", value.X, value.Y, value.Z); } 

template <typename T>
class HostGrid 
{
    std::unique_ptr<T[]> mData;
    Grid<T> mView;

public:
    HostGrid() = default;

    HostGrid(const size_t size, const T initValue) : HostGrid(size, size, size, initValue) {}

    HostGrid(const size_t sizeX, const size_t sizeY, const size_t sizeZ, const T initValue)
    {
        mData = std::make_unique<T[]>(sizeX * sizeY * sizeZ);
        mView = Grid(mData.get(), sizeX, sizeY, sizeZ);
        std::fill(mView.mGrid.begin(), mView.mGrid.end(), initValue);
    }

    HostGrid(const DeviceGrid<T>& device) 
    {
        const auto& v = device.View();
        mData = std::make_unique<T[]>(v.mSizeX * v.mSizeY * v.mSizeZ);
        mView = Grid<T>(mData.get(), v.mSizeX, v.mSizeY, v.mSizeZ);
        device.mData.CopyToHost(mData.get(), mView.Size());
    }

    HostGrid(const HostGrid& other) 
    {
        const auto& v = other.View();
        mData = std::make_unique<T[]>(v.mSizeX * v.mSizeY * v.mSizeZ);
        mView = Grid(mData.get(), v.mSizeX, v.mSizeY, v.mSizeZ);
        std::copy(v.mGrid.begin(), v.mGrid.end(), mView.mGrid.begin()); 
    }

    HostGrid(HostGrid&& other) { swap(other); }

    HostGrid& operator=(HostGrid other) { swap(other); return *this; }

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
    CudaPtr<T> mData;
    Grid<T> mView;

public:
    DeviceGrid() = default;

    DeviceGrid(const size_t size) : DeviceGrid(size, size, size) {}

    DeviceGrid(const size_t sizeX, const size_t sizeY, const size_t sizeZ)
    {
        const size_t storageSize = (sizeX * sizeY * sizeZ);

        mData = CudaPtr<T>(storageSize);
        mView = Grid(mData.get(), sizeX, sizeY, sizeZ);
    }

    DeviceGrid(const HostGrid<T>& host) 
    {
        const auto& v = host.View();
        const size_t storageSize = (v.mSizeX * v.mSizeY * v.mSizeZ);

        mData = CudaPtr(host.mData.get(), storageSize);
        mView = Grid(mData.get(), v.mSizeX, v.mSizeY, v.mSizeZ);
    }

    DeviceGrid(const DeviceGrid<T>& other) 
    {
        const auto& v = other.View();
        const size_t storageSize = (v.mSizeX * v.mSizeY * v.mSizeZ);
        
        mData = CudaPtr(other.mData);
        mView = Grid(mData.get(), v.mSizeX, v.mSizeY, v.mSizeZ);
    }

    DeviceGrid(DeviceGrid<T>&& other) { swap(other); }

    DeviceGrid& operator=(const DeviceGrid<T>& other) {
        if (this == &other) return *this;
        const auto& v = other.View();

        mData = other.mData;
        mView = Grid(mData.get(), v.SizeX(), v.SizeY(), v.SizeZ());
        return *this;
    }
    
    DeviceGrid& operator=(DeviceGrid<T>&& other) { swap(other); return *this; }

    void swap(DeviceGrid& other)
    {
        using std::swap;
        swap(mData, other.mData);
        swap(mView, other.mView);
    }

    friend void swap(DeviceGrid& first, DeviceGrid& second) { first.swap(second); }

    inline Grid<T>& View() { return mView; }

    inline const Grid<T>& View() const { return mView; }

    friend class HostGrid<T>;
};

#endif // !GRID_H
