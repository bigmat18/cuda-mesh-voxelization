/* CUDA kernel to test SIMT deadlock effect in pre-Volta and post-Volta GPUs */

#include<iostream>

__global__ void prova_kernel(volatile bool *readyA, volatile bool *readyB)
{
    if (threadIdx.x == 0) {
            ; // A;
            *readyA = true;
            while(!(*readyB));
    }
    else if (threadIdx.x == 1) {
            ; // B;
            *readyB = true;
            while(!(*readyA));
    }
    ; // C;
}

int main()
{
        bool *readyA_cpu;
        bool *readyA_gpu;
        bool *readyB_cpu;
        bool *readyB_gpu;
        cudaSetDevice(0); // set the working device
        readyA_cpu = (bool *) malloc(sizeof(bool));
        cudaMalloc(&readyA_gpu, sizeof(bool));
        *readyA_cpu = false;
        cudaMemcpy(readyA_gpu, readyA_cpu, sizeof(bool), cudaMemcpyHostToDevice);
        readyB_cpu = (bool *) malloc(sizeof(bool));
        cudaMalloc(&readyB_gpu, sizeof(bool));
        *readyB_cpu = false;
        cudaMemcpy(readyB_gpu, readyB_cpu, sizeof(bool), cudaMemcpyHostToDevice);
        prova_kernel<<<1, 2>>>(readyA_gpu, readyB_gpu);
        //gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        return 0;
}
