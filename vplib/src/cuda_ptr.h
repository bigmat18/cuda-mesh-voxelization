#ifndef CUDA_PTR_H
#define CUDA_PTR_H

#include <cstddef>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <memory>
#include <debug_utils.h>

template <typename T>
class CudaPtr {
    T* mPtr = nullptr;
    size_t mSize = 0;

public:
    CudaPtr() = default;

    CudaPtr(const size_t size) : mSize(size) 
    {
        gpuAssert(cudaMalloc((void**) &mPtr, mSize * sizeof(T)));
    }

    CudaPtr(const std::unique_ptr<T[]>& hostPtr, const size_t size) : mSize(size)
    {
        gpuAssert(cudaMalloc((void**) &mPtr, mSize * sizeof(T)));
        gpuAssert(cudaMemcpy(mPtr, hostPtr.get(), mSize * sizeof(T), cudaMemcpyHostToDevice));
    }

    CudaPtr(const CudaPtr<T>& other) : mSize(other.mSize)
    {
        gpuAssert(cudaMalloc((void**) &mPtr, mSize * sizeof(T)));
        gpuAssert(cudaMemcpy(mPtr, other.mPtr, mSize * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    CudaPtr(CudaPtr<T>&& other) { swap(other); }

    ~CudaPtr() { if(mPtr) cudaFree(mPtr); }

    CudaPtr& operator=(const CudaPtr& other)
    {
        if (this == &other) return *this;
    
        if(mSize != other.mSize) {
            if (mPtr) cudaFree(mPtr);
            mSize = other.mSize;
            gpuAssert(cudaMalloc((void**) &mPtr, mSize * sizeof(T)));
        }
        gpuAssert(cudaMemcpy(mPtr, other.mPtr, mSize * sizeof(T), cudaMemcpyDeviceToDevice));
        return *this;
    }

    CudaPtr& operator=(CudaPtr&& other) { swap(other); return *this; }

    void swap(CudaPtr<T>& other) 
    {
        using std::swap;
        swap(mPtr, other.mPtr);
        swap(mSize, other.mSize);
    }

    friend void swap(CudaPtr<T>& first, CudaPtr<T> second) { first.swap(second); }

    T* get() { return mPtr; }

    const T* get() const { return mPtr; }

    T& operator[](size_t idx) { return mPtr[idx]; }

    const T& operator[](size_t idx) const { return mPtr[idx]; }

    inline size_t Size() const { return mSize; }

    void copyFromHost(const T* src, const size_t size)
    {
        if(mSize != size) {
            if (mPtr) cudaFree(mPtr);
            mSize = size;
            gpuAssert(cudaMalloc((void**) &mPtr, mSize * sizeof(T)));
        }
        gpuAssert(cudaMemcpy(mPtr, src, mSize * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copyToHost(T* dst, const size_t size) const {
        if (size > mSize) 
            throw std::out_of_range("copyToHost: size too large");
        gpuAssert(cudaMemcpy(dst, mPtr, size * sizeof(T), cudaMemcpyDeviceToHost));
    }
};


#endif // !CUDA_PTR_H

