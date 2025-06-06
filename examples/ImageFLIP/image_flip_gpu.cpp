/* CUDA application computing the horizontal flip of a bitmap image */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "utils.h"

#define BLOCK_DIM 16 // blocks of BLOCK_DIM*BLOCKD_DIM threads

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

__global__ void image_flip(pel *ImgDst, pel *ImgSrc, int w, int h, int wbytes)
{
    unsigned int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int iy = (blockIdx.y * blockDim.y)+ threadIdx.y;
    if (ix < w && iy < h) {
        unsigned int offset_src = (iy * wbytes) + ix * 3;
        unsigned int ixx = w - ix - 1;
        unsigned int offset_dst = (iy * wbytes) + ixx * 3;
        ImgDst[offset_dst] = ImgSrc[offset_src];
        ImgDst[offset_dst + 1] = ImgSrc[offset_src + 1];
        ImgDst[offset_dst + 2] = ImgSrc[offset_src + 2];
    }
}

int main(int argc, char **argv)
{
    pel *imgSrc, *imgDst;
    pel *imgSrcGPU, *imgDstGPU;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <InputFilename> <OutputFilename>" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create host memory to store the input and output images
    struct img_Descr img_d_source;
    imgSrc = read_bmpFile(argv[1], &img_d_source);
    if (imgSrc == NULL) {
        std::cout << "Error reading input image" << std::endl;
        exit(EXIT_FAILURE);
    }
    imgDst = (pel *) malloc(img_d_source.rowByte * img_d_source.height);
    if (imgDst == NULL) {
        free(imgSrc);
        std::cout << "Error allocation destination image" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaSetDevice(0); // set the working device
    // Allocate device memory for the input and output images
    gpuErrchk(cudaMalloc((void**) &imgSrcGPU, img_d_source.rowByte * img_d_source.height));
    gpuErrchk(cudaMalloc((void**) &imgDstGPU, img_d_source.rowByte * img_d_source.height));

    uint64_t initial_time = current_time_nsecs();

    // Copy host memory buffers into device memory buffers
    gpuErrchk(cudaMemcpy(imgSrcGPU, imgSrc, img_d_source.rowByte * img_d_source.height, cudaMemcpyHostToDevice));

    // Invoke kernels (define grid and block sizes)
    int rowBlocks = std::ceil(((double) img_d_source.width) / BLOCK_DIM);
    int columnBlocks = std::ceil(((double) img_d_source.height) / BLOCK_DIM);
    dim3 gridDim(rowBlocks, columnBlocks);
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);

    uint64_t initial_time2 = current_time_nsecs();

    image_flip<<<gridDim, blockDim>>>(imgDstGPU, imgSrcGPU, img_d_source.width, img_d_source.height, img_d_source.rowByte);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    uint64_t end_time2 = current_time_nsecs();

    // Copy output (results) from GPU buffer to host (CPU) memory
    gpuErrchk(cudaMemcpy(imgDst, imgDstGPU, img_d_source.rowByte * img_d_source.height, cudaMemcpyDeviceToHost));

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed2 = end_time2 - initial_time2;
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Kernel time: " << ((float) elapsed2)/1000.0 << " usec" << std::endl;
    std::cout << "Elapsed time: " << ((float) elapsed)/1000.0 << " usec" << std::endl;

    // Write the flipped image back to disk
    write_bmpFile(imgDst, argv[2], &img_d_source);

    // Deallocate CPU, GPU memory
    cudaFree(imgSrcGPU);
    cudaFree(imgDstGPU);
    free(imgSrc);
    free(imgDst);   
    return 0;
}
