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
    row_ptr[0] = nnz;
    // iterate over the row of B
    for (int i = 0; i < n; i++) {
        // iterate over the non-zeros of i-th row
        for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++) {
            // column index of the non-zero in i-th row in dense matrix
            int j = B->col_ind[p];

            // search in the C matrix
            // iterate over the non-zeros of j-th row of C
            for (int k = C->row_ptr[j]; k < C->row_ptr[j+1]; k++) {
                // C->col_ind[k] === column index of the non-zero in j-th row in C dense matrix
                if (C->col_ind[k] == i) {
                    col_ind[nnz] = j;
                    val[nnz] = B->val[p] * C->val[k];
                    printf("i: %d\tj: %d\n", i, j);
                    printf("%.2f\n", val[nnz]);
                    nnz++;
                    // break;
                }
            }
        }
        row_ptr[i+1] = nnz;
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

void printCSRDense(const CSRMatrix *M) {
    if (!M) return;

    for (int i = 0; i < M->rows; i++) {
        int row_start = M->row_ptr[i];
        int row_end   = M->row_ptr[i + 1];

        // Pointer into current row's CSR entries
        int csr_index = row_start;

        for (int j = 0; j < M->cols; j++) {
            double val = 0.0;
            if (csr_index < row_end && M->col_ind[csr_index] == j) {
                val = M->val[csr_index];
                csr_index++;
            }
            printf("%6.2f ", val);
        }
        printf("\n");
    }
}

int main() {
    // Example: A = [ [1,0,2],
    //                [0,3,0],
    //                [4,0,5] ]
    CSRMatrix A;
    A.rows = 5; A.cols = 5; A.nnz = 6;
    int Arp[] = {0,2,4,4,6,6};
    int Aci[] = {2,4,2,3,1,2};
    double Aval[] = {3,4,5,7,2,6};
    // A.rows = 3; A.cols = 3; A.nnz = 5;
    // int Arp[] = {0,2,3,5};
    // int Aci[] = {0,2,1,0,2};
    // double Aval[] = {1,2,3,4,5};
    A.row_ptr = Arp; A.col_ind = Aci; A.val = Aval;

    // B = [ [5,0,0],
    //        [0,6,0],
    //        [7,0,8] ]
    CSRMatrix B;
    B.rows = 5; B.cols = 5; B.nnz = 6;
    int Brp[] = {0,0,2,4,6,6};
    int Bci[] = {2,4,2,3,1,2};
    double Bval[] = {3,4,5,7,2,6};
    // B.rows = 3; B.cols = 3; B.nnz = 4;
    // int Brp[] = {0,1,2,4};
    // int Bci[] = {0,1,0,2};
    // double Bval[] = {5,6,7,8};
    B.row_ptr = Brp; B.col_ind = Bci; B.val = Bval;

    CSRMatrix C = multiplyCSR(&A, &B);
    printCSR(&C);
    printCSRDense(&C);

    free(C.row_ptr);
    free(C.col_ind);
    free(C.val);
    return 0;
}