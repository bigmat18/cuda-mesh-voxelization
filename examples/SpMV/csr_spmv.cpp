/* SpMV implementation with CSR sparse matrices */

#include<iostream>
#include<stdlib.h>
#include "utility_sparse.hpp"

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

__global__ void csr_spmv(csr_matrix *A, float *b, float *c)
{
    unsigned int row = (blockIdx.x*blockDim.x) + threadIdx.x;
    if (row < A->numRows) {
        float sum = 0.0f;
        for (unsigned int i = A->rowPtr[row]; i < A->rowPtr[row+1]; i++) {
            unsigned int col = A->columnIdx[i];
            float value = A->value[i];
            sum += b[col]*value;
        }
        c[row] = sum;
    }
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <L> <B> <density>" << std::endl;
        exit(1);
    }

    int L = atoi(argv[1]);
    int B = atoi(argv[2]);
    double density = atof(argv[3]);

    // initialization of the sparse matrix (host side)
    csr_matrix host_A;
    gen_sparseMatrix_CSR(&host_A, L, density);

    // initialization of the dense vector (host side)
    float *host_b = (float *) malloc(L*sizeof(float));
    for (int i=0; i<L; i++) {
        host_b[i] = rand() % 100;
    }

    // initialization of the result vector (host side)
    float *host_c = (float *) malloc(L*sizeof(float));
    memset(host_c, 0, L*sizeof(float));

    cudaSetDevice(0); // set the working device

    uint64_t initial_time = current_time_nsecs();

    // initialization of the sparse matrix (device side)
    csr_matrix tmp_A;
    tmp_A.numRows = host_A.numRows;
    tmp_A.non_zeros = host_A.non_zeros;
    gpuErrchk(cudaMalloc((void**) &(tmp_A.rowPtr), (host_A.numRows+1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &(tmp_A.columnIdx), host_A.non_zeros*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &(tmp_A.value), host_A.non_zeros*sizeof(float)));
    gpuErrchk(cudaMemcpy(tmp_A.rowPtr, host_A.rowPtr, (host_A.numRows+1)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tmp_A.columnIdx, host_A.columnIdx, host_A.non_zeros*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tmp_A.value, host_A.value, host_A.non_zeros*sizeof(float), cudaMemcpyHostToDevice));
    csr_matrix *dev_A;
    gpuErrchk(cudaMalloc((void**) &dev_A, sizeof(csr_matrix)));
    gpuErrchk(cudaMemcpy(dev_A, &tmp_A, sizeof(csr_matrix), cudaMemcpyHostToDevice));

    // initialization of the dense vector (device side)
    float *dev_b;
    gpuErrchk(cudaMalloc((void**) &dev_b, L*sizeof(float)));
    gpuErrchk(cudaMemcpy(dev_b, host_b, L*sizeof(float), cudaMemcpyHostToDevice));

    // initialization of the result vector (device side)
    float *dev_c;
    gpuErrchk(cudaMalloc((void**) &dev_c, L*sizeof(float)));
    cudaMemset(dev_c, 0, L*sizeof(float));

    uint64_t initial_time2 = current_time_nsecs();

    // Perform computation on GPU
    unsigned int numBlocks = std::ceil(((double) host_A.numRows) / B);
    csr_spmv<<<numBlocks, B>>>(dev_A, dev_b, dev_c);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    uint64_t end_time2 = current_time_nsecs();
    uint64_t elapsed2 = end_time2 - initial_time2;
    std::cout << "Kernel time: " << ((float) elapsed2)/1000.0 << " usec" << std::endl;

    // Copy results from GPU memory
    gpuErrchk(cudaMemcpy(host_c, dev_c, L*sizeof(float), cudaMemcpyDeviceToHost));

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Elapsed time: " << ((float) elapsed)/1000.0 << " usec" << std::endl;

    // Deallocate GPU memory
    gpuErrchk(cudaFree(tmp_A.rowPtr));
    gpuErrchk(cudaFree(tmp_A.columnIdx));
    gpuErrchk(cudaFree(tmp_A.value));
    gpuErrchk(cudaFree(dev_A));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(dev_c));

    // Check the results
    float *check_c = (float *) malloc(L*sizeof(float));
    memset(check_c, 0, L*sizeof(float));

    for (int i=0; i<host_A.numRows; i++) {
        float sum = 0.0f;
        for (int j=host_A.rowPtr[i]; j<host_A.rowPtr[i+1]; j++) {
            int col = host_A.columnIdx[j];
            float value = host_A.value[j];
            sum += host_b[col]*value;
        }
        check_c[i] = sum;
    }

    for (int i=0; i<L; i++) {
        if (check_c[i] != host_c[i]) {
            std::cout << "Result error!" << std::endl;
            abort();
        }
    }
    std::cout << "Result is ok!" << std::endl;

    // Deallocate host memory
    free(host_A.rowPtr);
    free(host_A.columnIdx);
    free(host_A.value);
    free(host_b);
    free(host_c);
    free(check_c);
    return 0;
}
