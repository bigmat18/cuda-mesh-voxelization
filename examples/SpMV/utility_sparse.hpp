/* Utility functions for the SpMV kernels */

#include<cassert>
#include<iostream>
#include<stdlib.h>

#ifndef UTILITY_SPARSE_H
#define UTILITY_SPARSE_H

struct coo_matrix
{
    int non_zeros;
    int *rowIdx;
    int *columnIdx;
    float *value;
};

struct csr_matrix
{
    int numRows;
    int non_zeros;
    int *rowPtr;
    int *columnIdx;
    float *value;
};

struct ell_matrix
{
    int numRows;
    int max_nz_row;
    int *non_zeros;
    int *columnIdx;
    float *value;
};

// Fisher-Yates algorithm
void shuffle(int *array, int size)
{
    for (int i=size - 1; i>0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// generate a random sparse matrix in host memory (COO format)
void gen_sparseMatrix_COO(coo_matrix *A, int L, double density)
{
    assert((density > 0) && (density < 1));
    A->non_zeros = (int) (L * L * density);
    assert(A->non_zeros>0);
    A->rowIdx = (int *) malloc(sizeof(int) * A->non_zeros);
    A->columnIdx = (int *) malloc(sizeof(int) * A->non_zeros);
    A->value = (float *) malloc(sizeof(float) * A->non_zeros);
    int *coords = (int *) malloc(sizeof(int) * L * L);
    for (int i=0; i<L*L; i++) {
        coords[i] = i;
    }
    shuffle(coords, L*L);
    for (int i=0; i<A->non_zeros; i++) {
        A->rowIdx[i] = coords[i] / L;
        A->columnIdx[i] = coords[i] % L;
        A->value[i] = rand() % 100;
    }
    free(coords);
}

// generate a random sparse matrix in host memory (CSR format)
void gen_sparseMatrix_CSR(csr_matrix *A, int L, double density)
{
    assert((density > 0) && (density < 1));
    A->non_zeros = (int) (L * L * density);
    A->numRows = L;
    A->rowPtr = (int *) malloc((A->numRows + 1) * sizeof(int));
    A->columnIdx = (int *) malloc(A->non_zeros * sizeof(int));
    A->value = (float *) malloc(A->non_zeros * sizeof(float));
    int nnz = 0;
    (A->rowPtr)[0] = 0;
    for (int i=0; i<L; i++) {
        for (int j=0; j<L; j++) {
            if (((float) rand() / RAND_MAX < density) && (nnz < A->non_zeros)) {
                (A->columnIdx)[nnz] = j;
                (A->value)[nnz] = rand() % 100;
                nnz++;
            }
        }
        (A->rowPtr)[i + 1] = nnz;
    }
    A->non_zeros = nnz;
}

// generate a random sparse matrix in host memory (ELL format)
void gen_sparseMatrix_ELL(ell_matrix *A, int L, double density)
{
    assert((density > 0) && (density < 1));
    A->numRows = L;
    A->non_zeros = (int *) malloc(L * sizeof(int));
    A->max_nz_row = (int)(L * density); // max number of non-zero elements in the same row
    A->columnIdx = (int *) malloc(L * A->max_nz_row * sizeof(int));
    A->value = (float *) malloc(L * A->max_nz_row * sizeof(float));
    int row_with_max_non_zeros = rand() % L;
    for (int i=0; i<L; i++) {
        if (i != row_with_max_non_zeros) {
            A->non_zeros[i] = rand() % (A->max_nz_row + 1);
        }
        else {
            A->non_zeros[i] = A->max_nz_row;
        }
        int *c = (int *) malloc(L * sizeof(int));
        for (int k=0; k<L; k++) {
            c[k] = k;
        }
        shuffle(c, L);
        for (int j=0; j<A->max_nz_row; j++) {
            int pos = (j*A->numRows) + i;
            if (j < A->non_zeros[i]) {
                A->columnIdx[pos] = c[j];
                A->value[pos] = rand() % 100;
            }
            else {
                A->columnIdx[pos] = -1; // padding
                A->value[pos] = 0.0f; // padding
            }
        }
        free(c);
    }
}

#endif
