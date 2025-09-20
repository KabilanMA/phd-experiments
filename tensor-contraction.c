#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int rows;
    int cols;
    int nnz;       // number of non-zeros
    int *row_ptr;  // size = rows + 1
    int *col_ind;  // size = nnz
    double *val;   // size = nnz
} CSRMatrix;

void freeCSRMatrix(CSRMatrix *M) {
    if (!M) return;  // null check

    if (M->row_ptr) {
        free(M->row_ptr);
        M->row_ptr = NULL;
    }
    if (M->col_ind) {
        free(M->col_ind);
        M->col_ind = NULL;
    }
    if (M->val) {
        free(M->val);
        M->val = NULL;
    }

    // Reset metadata
    M->rows = 0;
    M->cols = 0;
    M->nnz = 0;
}

// Multiply A = B * C (element wise row-col multiplication)
CSRMatrix multiplyCSR(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;
    int *row_ptr = calloc(n + 1, sizeof(int));
    int capacity = B->nnz > C->nnz ? B->nnz : C->nnz;
    int *col_ind = malloc(capacity * sizeof(int));
    double *val = malloc(capacity * sizeof(double));

    int nnz = 0;
    for (int i = 0; i < n; i++) {
        // if (B->row_ptr[i+1] <= B->row_ptr[i]) continue;
        for (int j = B->row_ptr[i]; j < B->row_ptr[i+1]; j++) {

            // search in the C matrix
            for (int k = 0; k < m; i++) {
                
            }
        }
        for (int j = 0; j < m; i++) {

        }
        // Temporary dense row (naive approach)
        double *accum = calloc(n, sizeof(double));

        for (int jj = B->row_ptr[i]; jj < B->row_ptr[i + 1]; jj++) {
            int a_col = B->col_ind[jj];
            double a_val = B->val[jj];

            // multiply row i of A with row a_col of B
            for (int kk = C->row_ptr[a_col]; kk < B->row_ptr[a_col + 1]; kk++) {
                int b_col = B->col_ind[kk];
                double b_val = B->val[kk];
                accum[b_col] += a_val * b_val;
            }
        }

        // store results into C
        row_ptr[i] = nnz;
        for (int j = 0; j < n; j++) {
            if (accum[j] != 0.0) {
                if (nnz >= capacity) {
                    capacity *= 2;
                    col_ind = realloc(col_ind, capacity * sizeof(int));
                    val = realloc(val, capacity * sizeof(double));
                }
                col_ind[nnz] = j;
                val[nnz] = accum[j];
                nnz++;
            }
        }
        free(accum);
    }
    row_ptr[n] = nnz;

    CSRMatrix A = {B->rows, C->cols, nnz, row_ptr, col_ind, val};
    return A;
}

// Utility to print CSR matrix
void printCSR(const CSRMatrix *M) {
    printf("CSR Matrix: %d x %d, nnz = %d\n", M->rows, M->cols, M->nnz);
    printf("row_ptr: ");
    for (int i = 0; i <= M->rows; i++) printf("%d ", M->row_ptr[i]);
    printf("\ncol_ind: ");
    for (int i = 0; i < M->nnz; i++) printf("%d ", M->col_ind[i]);
    printf("\nval:     ");
    for (int i = 0; i < M->nnz; i++) printf("%.2f ", M->val[i]);
    printf("\n");
}