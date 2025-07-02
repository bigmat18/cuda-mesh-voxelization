#ifndef VOXELS_GRID
#define VOXELS_GRID

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <sys/types.h>
#include <type_traits>
#include <cassert>
#include <debug_utils.h>
#include <cuda_runtime.h>

template<typename T, typename... Types>
__device__ constexpr bool is_one_of = 
    ( std::is_same_v<T, Types> || ... );

template <
    typename T = uint8_t,
    typename = std::enable_if_t<is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>>>
class HostVoxelsGrid;

template <
    typename T = uint32_t,
    typename = std::enable_if_t<is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>>>
class DeviceVoxelsGrid;


template <
    typename T = uint32_t,
    bool device = false,
    typename = std::enable_if_t<is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>>>
class VoxelsGrid 
{
    std::span<T> mVoxels;
    size_t mVoxelsPerSide;
    float mSideLength; 
    float mOriginX = 0;
    float mOriginY = 0;
    float mOriginZ = 0;

    class Bit {
        T* mWord;
        T mMask;

    public:
 
        __host__ __device__
        Bit(T* word, T mask) :
            mWord(word), mMask(mask) {}

        __host__ __device__
        Bit& operator=(bool value) 
        {
            if constexpr (device) {
                static_assert(sizeof(T) >= 4);
                if(value) atomicOr(mWord, mMask);
                else      atomicAnd(mWord, ~mMask);
            } else {
                if(value) (*mWord) |= mMask;
                else      (*mWord) &= ~mMask;
            }
            return *this;
        }

        __host__ __device__
        Bit& operator^=(bool value)
        {
            if constexpr (device) {
               static_assert(sizeof(T) >= 4);
               if(value) atomicXor(mWord, mMask);
            } else {
                if(value) (*mWord) ^= mMask;
            }
            return *this;
        }

        __host__ __device__
        operator bool() const { return ((*mWord) & mMask) != 0; }
    };

public:

    VoxelsGrid() = default;

    __host__ __device__
    VoxelsGrid(T* data, const size_t voxelsPerSide, const float sideLength) :
        mVoxelsPerSide(voxelsPerSide), mSideLength(sideLength) 
    {
        mVoxels = std::span<T>(data, StorageSize(mVoxelsPerSide));
    }

    __host__ __device__
    Bit operator() (size_t x, size_t y, size_t z) {
        assert(x < mVoxelsPerSide); 
        assert(y < mVoxelsPerSide); 
        assert(z < mVoxelsPerSide); 

        size_t index = Index(x, y, z);
        return Bit(&mVoxels[index / WordSize()], (T(1) << (index % WordSize())));
    }

    __host__ __device__
    bool operator()(size_t x, size_t y, size_t z) const {
        assert(x < mVoxelsPerSide); 
        assert(y < mVoxelsPerSide); 
        assert(z < mVoxelsPerSide); 

        size_t index = Index(x, y, z);
        return (mVoxels[index / WordSize()] & (T(1) << (index % WordSize()))) != 0;
    }

    __host__ __device__
    void XorWord(size_t x, size_t y, size_t z, T word)
    {
        assert(x < mVoxelsPerSide); 
        assert(y < mVoxelsPerSide); 
        assert(z < mVoxelsPerSide); 

        size_t index = Index(x, y, z);
        if constexpr (device) {
            atomicXor(&mVoxels[index / WordSize()], word);
        } else {
            mVoxels[index / WordSize()] |= word;
        }
    }

    template <typename fun> 
    __host__ __device__
    void SetWord(size_t x, size_t y, size_t z, T word, fun op)
    {
        assert(x < mVoxelsPerSide); 
        assert(y < mVoxelsPerSide); 
        assert(z < mVoxelsPerSide); 

        size_t index = Index(x, y, z);
        op(mVoxels[index / WordSize()], word);
    }

    __host__ __device__
    T GetWord(size_t x, size_t y, size_t z) const
    {

        assert(x < mVoxelsPerSide); 
        assert(y < mVoxelsPerSide); 
        assert(z < mVoxelsPerSide); 

        size_t index = Index(x, y, z);
        return mVoxels[index / WordSize()];
    }

    __host__ __device__
    void SetOrigin(float x, float y, float z) { mOriginX = x; mOriginY = y; mOriginZ = z; }

    __host__ __device__
    inline size_t Index(size_t x, size_t y, size_t z) const { return (z * mVoxelsPerSide * mVoxelsPerSide) + (y * mVoxelsPerSide) + x; }

    // The size of a single word, the voxels space are stored in block of uint (8, 16, 32 ... bit)
    __host__ __device__
    static inline size_t WordSize() { return sizeof(T) * 8; }

    // The size of alla voxels space (mVoxelsPerSize * mVoxelsPerSize * mVoxelsPerSize)
    __host__ __device__
    inline size_t SpaceSize() const { return mVoxelsPerSide * mVoxelsPerSide * mVoxelsPerSide; }

    // The size of a side of voxels grid.
    __host__ __device__
    inline size_t VoxelsPerSide() const { return mVoxelsPerSide; }

    // The size of single voxel. Understood as the side of square
    __host__ __device__
    inline float VoxelSize() const { return mSideLength / mVoxelsPerSide; }

    // Length of a size in terms of real scale
    __host__ __device__
    inline float SideLength() const { return mSideLength; }

    // How much word we need to store with a specific size
    __host__ __device__
    static inline size_t StorageSize(const size_t sideSize) { 
        return ((sideSize * sideSize * sideSize) + (WordSize() - 1)) / WordSize();
    }

    // Get Origin X value of grid
    __host__ __device__
    inline float OriginX() const { return mOriginX; }

    // Get Origin Y value of grid
    __host__ __device__
    inline float OriginY() const { return mOriginY; }

    // Get Origin Z value of grid
    __host__ __device__
    inline float OriginZ() const { return mOriginZ; }

    inline void Print() const {
        for(int z = 0; z <= mVoxelsPerSide; ++z) {
            for (int y = 0; y <= mVoxelsPerSide; ++y) {
                for (int x = 0; x <= mVoxelsPerSide; ++x) {
                    printf("%d ", (*this)(x, y, z));    
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    friend class HostVoxelsGrid<T>;
    friend class DeviceVoxelsGrid<T>;
};


template <typename T, typename>
class HostVoxelsGrid 
{
    std::unique_ptr<T[]> mData;
    VoxelsGrid<T, false> mView;

public:
    HostVoxelsGrid(const size_t voxelsPerSide, const float sideLength)
    {
        mData = std::make_unique<T[]>(VoxelsGrid<T>::StorageSize(voxelsPerSide));
        mView = VoxelsGrid<T, false>(mData.get(), voxelsPerSide, sideLength);
        std::fill(mView.mVoxels.begin(), mView.mVoxels.end(), 0);
    }

    HostVoxelsGrid(const DeviceVoxelsGrid<T>& device) 
    {
        const size_t storageSize = VoxelsGrid<T>::StorageSize(device.View().mVoxelsPerSide);
        mData = std::make_unique<T[]>(storageSize);

        mView = VoxelsGrid<T>(mData.get(), device.View().mVoxelsPerSide, device.View().mSideLength);
        gpuAssert(cudaMemcpy(mData.get(), device.mData, storageSize * sizeof(T), cudaMemcpyDeviceToHost));
        mView.SetOrigin(device.View().OriginX(), device.View().OriginY(), device.View().OriginZ());
    }

    HostVoxelsGrid(const HostVoxelsGrid&) = delete;
    HostVoxelsGrid& operator=(const HostVoxelsGrid&) = delete;

    inline VoxelsGrid<T, false>& View() { return mView; }

    inline const VoxelsGrid<T, false>& View() const { return mView; }

    friend class DeviceVoxelsGrid<T>;
};


template <typename T, typename>
class DeviceVoxelsGrid 
{
    T* mData = nullptr;
    VoxelsGrid<T, true> mView;

public:
    DeviceVoxelsGrid(const size_t voxelsPerSide, const float sideLength)
    {
        const size_t storageSize = VoxelsGrid<T>::StorageSize(voxelsPerSide) * sizeof(T);           
        gpuAssert(cudaMalloc((void**) &mData, storageSize));   
        gpuAssert(cudaMemset(mData, 0, storageSize));
        mView = VoxelsGrid<T, true>(mData, voxelsPerSide, sideLength);
    }

    DeviceVoxelsGrid(const HostVoxelsGrid<T>& host) 
    {
        const size_t storageSize = VoxelsGrid<T>::StorageSize(host.View().mVoxelsPerSide) * sizeof(T);
        gpuAssert(cudaMalloc((void**) &mData, storageSize));
        mView = VoxelsGrid<T, true>(mData, host.View().mVoxelsPerSide, host.View().mSideLength);
        gpuAssert(cudaMemcpy(mData, host.mData.get(), storageSize, cudaMemcpyHostToDevice));
        mView.SetOrigin(host.View().OriginX(), host.View().OriginY(), host.View().OriginZ());
    }

    ~DeviceVoxelsGrid()
    {
        if(mData)   
            gpuAssert(cudaFree(mData));
    }   

    DeviceVoxelsGrid(const DeviceVoxelsGrid&) = delete;
    DeviceVoxelsGrid& operator=(const DeviceVoxelsGrid&) = delete;

    inline VoxelsGrid<T, true>& View() { return mView; }

    inline const VoxelsGrid<T, true>& View() const { return mView; }

    friend class HostVoxelsGrid<T>;
};


using VoxelsGrid8bit  = VoxelsGrid<uint8_t>;
using VoxelsGrid16bit = VoxelsGrid<uint16_t>;
using VoxelsGrid32bit = VoxelsGrid<uint32_t>;
using VoxelsGrid64bit = VoxelsGrid<uint64_t>;


using VoxelsGrid8bitHost  = VoxelsGrid<uint8_t, false>;
using VoxelsGrid16bitHost = VoxelsGrid<uint16_t, false>;
using VoxelsGrid32bitHost = VoxelsGrid<uint32_t, false>;
using VoxelsGrid64bitHost = VoxelsGrid<uint64_t, false>;

using HostVoxelsGrid8bit  = HostVoxelsGrid<uint8_t>;
using HostVoxelsGrid16bit = HostVoxelsGrid<uint16_t>;
using HostVoxelsGrid32bit = HostVoxelsGrid<uint32_t>;
using HostVoxelsGrid64bit = HostVoxelsGrid<uint64_t>;


using VoxelsGrid32bitDev = VoxelsGrid<uint32_t, true>;
using VoxelsGrid64bitDev = VoxelsGrid<uint64_t, true>;    

using DeviceVoxelsGrid32bit = DeviceVoxelsGrid<uint32_t>;
using DeviceVoxelsGrid64bit = DeviceVoxelsGrid<uint64_t>;

#endif // !VOXELS_GRID
