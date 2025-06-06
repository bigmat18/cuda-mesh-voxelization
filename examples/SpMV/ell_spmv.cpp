/* SpMV implementation with ELL sparse matrices */

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

__global__ void ell_spmv(ell_matrix *A, float *b, float *c)
{
    unsigned int row = (blockIdx.x*blockDim.x) + threadIdx.x;
    if (row < A->numRows) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < A->non_zeros[row]; i++) {
            unsigned int pos = (i*A->numRows) + row;
            unsigned int col = A->columnIdx[pos];
            float value = A->value[pos];
            sum += b[col] * value;
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
    ell_matrix host_A;
    gen_sparseMatrix_ELL(&host_A, L, density);

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
    ell_matrix tmp_A;
    tmp_A.numRows = host_A.numRows;
    gpuErrchk(cudaMalloc((void**) &(tmp_A.non_zeros), host_A.numRows*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &(tmp_A.columnIdx), host_A.numRows*host_A.max_nz_row*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &(tmp_A.value), host_A.numRows*host_A.max_nz_row*sizeof(float)));
    gpuErrchk(cudaMemcpy(tmp_A.non_zeros, host_A.non_zeros, host_A.numRows*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tmp_A.columnIdx, host_A.columnIdx, host_A.numRows*host_A.max_nz_row*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(tmp_A.value, host_A.value, host_A.numRows*host_A.max_nz_row*sizeof(float), cudaMemcpyHostToDevice));
    ell_matrix *dev_A;
    gpuErrchk(cudaMalloc((void**) &dev_A, sizeof(ell_matrix)));
    gpuErrchk(cudaMemcpy(dev_A, &tmp_A, sizeof(ell_matrix), cudaMemcpyHostToDevice));

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
    ell_spmv<<<numBlocks, B>>>(dev_A, dev_b, dev_c);

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
    gpuErrchk(cudaFree(tmp_A.non_zeros));
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
        for (unsigned int j=0; j<host_A.non_zeros[i]; j++) {
            unsigned int pos = (j*host_A.numRows) + i;
            unsigned int col = host_A.columnIdx[pos];
            float value = host_A.value[pos];
            sum += host_b[col] * value;
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
    free(host_A.non_zeros);
    free(host_A.columnIdx);
    free(host_A.value);
    free(host_b);
    free(host_c);
    free(check_c);
    return 0;
}
