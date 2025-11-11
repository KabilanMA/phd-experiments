#include "raw_kernel.hpp"

using namespace std;

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
// A = einsum("ij,ji->ij", B, C)
static CSRMatrix* _raw_kernel_1_1(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = B->nnz > C->nnz ? B->nnz : C->nnz;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

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
                    // printf("i: %d\tj: %d\n", i, j);
                    // printf("%.2f\n", val[nnz]);
                    nnz++;
                    break;
                }
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
static CSRMatrix* _raw_kernel_1_2(const CSRMatrix *B, const CSCMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = B->nnz > C->nnz ? B->nnz : C->nnz;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    int nnz = 0;
    row_ptr[0] = nnz;
    // iterate over the row of B
    for (int i = 0; i < n; i++) {
        // iterate over the non-zeros of i-th row
        for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++) {
            // column index of the non-zero in i-th row in dense matrix
            int j = B->col_ind[p];

            for (int k = C->col_ptr[i]; k < C->col_ptr[i+1]; k++)
            {
                if (C->row_ind[k] == j) {
                    col_ind[nnz] = j;
                    val[nnz] = B->val[p] * C->val[k];
                    nnz++;
                    break;
                }
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
// A = einsum("ij,ji->ij", B, C)
static CSRMatrix* _raw_kernel_1_3(const CSRMatrix *B, const COOMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = B->nnz > C->values.size() ? B->nnz : C->values.size();
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

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
            for (int k = 0; k < C->row_indices.size(); k++)
            {
                if (C->row_indices[k] == j && C->col_indices[k] == i)
                {
                    col_ind[nnz] = j;
                    val[nnz] = B->val[p] * C->values[k];
                    // printf("i: %d\tj: %d\n", i, j);
                    // printf("%.2f\n", val[nnz]);
                    nnz++;
                    break;
                }
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
// A = einsum("ij,ji->ij", B, C)
static CSRMatrix* _raw_kernel_1_7(const COOMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int n = B->rows;
    int m = B->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = B->values.size() > C->nnz ? B->values.size() : C->nnz;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    int nnz = 0;
    row_ptr[0] = nnz;
    // iterate over the nnz of B
    for (int p = 0; p < B->values.size(); p++)
    {
        int i = B->row_indices[p];
        int j = B->col_indices[p];

        // search in the C matrix
        for (int k = C->row_ptr[j]; k < C->row_ptr[j+1]; k++) {
            if (C->col_ind[k] == i) {
                col_ind[nnz] = j;
                val[nnz] = B->values[p] * C->val[k];
                nnz++;
                break;
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
// A = einsum("ij,ji->ij", B, C)
static CSRMatrix* _raw_kernel_1_8(const COOMatrix *B, const CSCMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int n = B->rows;
    int m = B->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = B->values.size() > C->nnz ? B->values.size() : C->nnz;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    int nnz = 0;
    row_ptr[0] = nnz;
    // iterate over the nnz of B
    for (int p = 0; p < B->values.size(); p++)
    {
        int i = B->row_indices[p];
        int j = B->col_indices[p];

        // search in the C matrix
        for (int k = C->col_ptr[i]; k < C->col_ptr[i+1]; k++)
        {
            if (C->row_ind[k] == j) {
                col_ind[nnz] = j;
                val[nnz] = B->values[p] * C->val[k];
                nnz++;
                break;
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
// A = einsum("ij,ji->ij", B, C)
static CSRMatrix* _raw_kernel_1_9(const COOMatrix *B, const COOMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int n = B->rows;
    int m = B->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = B->values.size() > C->values.size() ? B->values.size() : C->values.size();
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    int nnz = 0;
    row_ptr[0] = nnz;
    // iterate over the nnz of B
    for (int p = 0; p < B->values.size(); p++)
    {
        int i = B->row_indices[p];
        int j = B->col_indices[p];

        // search in the C matrix
        for (int k = 0; k < C->values.size(); k++)
        {
            if (C->row_indices[k] == j && C->col_indices[k] == i)
            {
                col_ind[nnz] = j;
                val[nnz] = B->values[p] * C->values[k];
                nnz++;
                break;
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
static CSCMatrix* _raw_kernel_1_13(const CSCMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int nnz = 0;
    int n = B->rows;
    int m = B->cols;
    int *col_ptr = (int *)calloc(m+1, sizeof(int));
    int capacity = B->nnz > C->nnz ? B->nnz : C->nnz;
    int *row_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    col_ptr[0] = nnz;
    // iterate over the col of B
    for (int j = 0; j < m; j++)
    {
        // iterate over the non-zeros of the j-th column
        for (int p = B->col_ptr[j]; p < B->col_ptr[j+1]; p++)
        {
            int i = B->row_ind[p];

            // search in the C matrix
            // iterate through the non-zeros of the j-th column in C
            for (int k = C->row_ptr[j]; k < C->row_ptr[j+1]; k++)
            {
                if (C->col_ind[k] == i)
                {
                    row_ind[nnz] = j;
                    val[nnz] = B->val[p] * C->val[k];
                    nnz++;
                    break;
                }
            }
        }
        col_ptr[j+1] = nnz;
    }
    col_ptr[m] = nnz;

    CSCMatrix* A = (CSCMatrix*)malloc(sizeof(CSCMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->col_ptr = col_ptr;
    A->row_ind = row_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
static CSCMatrix* _raw_kernel_1_14(const CSCMatrix *B, const CSCMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int nnz = 0;
    int n = B->rows;
    int m = B->cols;
    int *col_ptr = (int *)calloc(m+1, sizeof(int));
    int capacity = B->nnz > C->nnz ? B->nnz : C->nnz;
    int *row_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    col_ptr[0] = nnz;
    // iterate over the col of B
    for (int j = 0; j < m; j++)
    {
        // iterate over the non-zeros of the j-th column
        for (int p = B->col_ptr[j]; p < B->col_ptr[j+1]; p++)
        {
            int i = B->row_ind[p];

            // search in the C matrix
            // iterate through the non-zeros of the j-th row in C
            for (int k = C->col_ptr[i]; k < C->col_ptr[i+1]; k++)
            {
                if (C->row_ind[k] == j) {
                    row_ind[nnz] = j;
                    val[nnz] = B->val[p] * C->val[k];
                    nnz++;
                    break;
                }
            }
        }
        col_ptr[j+1] = nnz;
    }
    col_ptr[m] = nnz;

    CSCMatrix* A = (CSCMatrix*)malloc(sizeof(CSCMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->col_ptr = col_ptr;
    A->row_ind = row_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
static CSCMatrix* _raw_kernel_1_15(const CSCMatrix *B, const COOMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int nnz = 0;
    int n = B->rows;
    int m = B->cols;
    int *col_ptr = (int *)calloc(m+1, sizeof(int));
    int capacity = B->nnz > C->values.size() ? B->nnz : C->values.size();
    int *row_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    col_ptr[0] = nnz;
    // iterate over the col of B
    for (int j = 0; j < m; j++)
    {
        // iterate over the non-zeros of the j-th column
        for (int p = B->col_ptr[j]; p < B->col_ptr[j+1]; p++)
        {
            int i = B->row_ind[p];

            // search in the C matrix
            // iterate through the non-zeros of the j-th row in C
            for (int k = 0; k < C->row_indices.size(); k++)
            {
                if (C->row_indices[k] == j && C->col_indices[k] == i)
                {
                    row_ind[nnz] = j;
                    val[nnz] = B->val[p] * C->values[k];
                    // printf("i: %d\tj: %d\n", i, j);
                    // printf("%.2f\n", val[nnz]);
                    nnz++;
                    break;
                }
            }
        }
        col_ptr[j+1] = nnz;
    }
    col_ptr[m] = nnz;

    CSCMatrix* A = (CSCMatrix*)malloc(sizeof(CSCMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->col_ptr = col_ptr;
    A->row_ind = row_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
static CSCMatrix* _raw_kernel_1_24(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int nnz = 0;
    int n = B->rows;
    int m = B->cols;
    int *col_ptr = (int *)calloc(m+1, sizeof(int));
    int capacity = B->nnz > C->nnz ? B->nnz : C->nnz;
    int *row_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    col_ptr[0] = nnz;
    // iterate over the column of B
    for (int j = 0; j < B->cols; j++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++)
            {
                if (B->col_ind[p] == j)
                {
                    // search C
                    for (int k = C->row_ptr[j]; k < C->row_ptr[j+1]; k++)
                    {
                        if (C->col_ind[k] == i) {
                            row_ind[nnz] = j;
                            val[nnz] = B->val[p] * C->val[k];
                            nnz++;
                            break;
                        }
                    }
                }
            }
        }
        col_ptr[j+1] = nnz;
    }
    col_ptr[m] = nnz;

    CSCMatrix* A = (CSCMatrix*)malloc(sizeof(CSCMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->col_ptr = col_ptr;
    A->row_ind = row_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
static CSCMatrix* _raw_kernel_1_25(const CSRMatrix *B, const CSCMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int nnz = 0;
    int n = B->rows;
    int m = B->cols;
    int *col_ptr = (int *)calloc(m+1, sizeof(int));
    int capacity = B->nnz > C->nnz ? B->nnz : C->nnz;
    int *row_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    col_ptr[0] = nnz;
    // iterate over the column of B
    for (int j = 0; j < B->cols; j++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++)
            {
                if (B->col_ind[p] == j)
                {
                    // search C
                    for (int k = C->col_ptr[i]; k < C->col_ptr[i+1]; k++) {
                        if (C->row_ind[k] == j) {
                            row_ind[nnz] = j;
                            val[nnz] = B->val[p] * C->val[k];
                            nnz++;
                            break;
                        }
                    }
                }
            }
        }
        col_ptr[j+1] = nnz;
    }
    col_ptr[m] = nnz;

    CSCMatrix* A = (CSCMatrix*)malloc(sizeof(CSCMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->col_ptr = col_ptr;
    A->row_ind = row_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
static CSCMatrix* _raw_kernel_1_26(const CSRMatrix *B, const COOMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int nnz = 0;
    int n = B->rows;
    int m = B->cols;
    int *col_ptr = (int *)calloc(m+1, sizeof(int));
    int capacity = B->nnz > C->values.size() ? B->nnz : C->values.size();
    int *row_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    col_ptr[0] = nnz;
    // iterate over the column of B
    for (int j = 0; j < B->cols; j++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++)
            {
                if (B->col_ind[p] == j)
                {
                    // search C
                    for (int k = 0; k < C->row_indices.size(); k++) 
                    {
                        if (C->row_indices[k] == j && C->col_indices[k] == i) {
                            row_ind[nnz] = j;
                            val[nnz] = B->val[p] * C->values[k];
                            nnz++;
                            break;
                        }
                    }
                }
            }
        }
        col_ptr[j+1] = nnz;
    }
    col_ptr[m] = nnz;

    CSCMatrix* A = (CSCMatrix*)malloc(sizeof(CSCMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->col_ptr = col_ptr;
    A->row_ind = row_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
static CSRMatrix* _raw_kernel_1_33(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = B->nnz > C->nnz ? B->nnz : C->nnz;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    int nnz = 0;
    row_ptr[0] = nnz;
    // iterate over the cols of C
    for (int j = 0; j < C->cols; j++)
    {
        for (int i = 0; i < C->rows; i++)
        {
            for (int k = C->row_ptr[i]; k < C->row_ptr[i+1]; k++)
            {
                if (C->col_ind[k] == j)
                {
                    // search in B
                    for (int p = B->row_ptr[j]; p < B->row_ptr[j+1]; p++)
                    {
                        if (B->col_ind[p] == i)
                        {
                            col_ind[nnz] = i;
                            val[nnz] = B->val[p] * C->val[k];
                            // std::cout << "val[nnz] = B->val[p] * C->val[k] ==>" << val[nnz] << " = " << B->val[p] << " * " << C->val[k] << std::endl;
                            // std::cout << "C->val[" << k << "]" << " ==> " << C->val[k] << std::endl;
                            nnz++;
                        }
                    }
                }
            }
        }
        row_ptr[j+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = B->cols;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
static CSRMatrix* _raw_kernel_1_34(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int n = B->rows;
    int m = B->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = B->nnz > C->nnz ? B->nnz : C->nnz;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    int nnz = 0;
    row_ptr[0] = nnz;
    // iterate over the rows of C
    for (int i = 0; i < C->rows; i++)
    {
        for (int p = C->row_ptr[i]; p < C->row_ptr[i+1]; p++)
        {
            int j = C->col_ind[p];
            
            // search in B
            for (int k = B->row_ptr[j]; k < B->row_ptr[j+1]; k++)
            {
                if (B->col_ind[k] == i)
                {
                    col_ind[nnz] = i;
                    val[nnz] = B->val[k] * C->val[p];
                    nnz++;
                    break;
                }
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
static CSRMatrix* _raw_kernel_1_35(const CSRMatrix *B, const CSCMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch!\n");
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = B->nnz > C->nnz ? B->nnz : C->nnz;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));

    int nnz = 0;
    row_ptr[0] = nnz;
    // iterate over the cols of C
    for (int j = 0; j < C->cols; j++)
    {
        for (int p = C->col_ptr[j]; p < C->col_ptr[j+1]; p++)
        {
            int i = C->row_ind[p];
            
            // search B
            for (int k = B->row_ptr[j]; k < B->row_ptr[j+1]; k++)
            {
                if (B->col_ind[k] == i)
                {
                    col_ind[nnz] = i;
                    val[nnz] = B->val[k] * C->val[p];
                    nnz++;
                    break;
                }
            }
        }
        row_ptr[j+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = B->cols;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

static int int_cmp(const void *a, const void *b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return (ia < ib) ? -1 : (ia > ib);
}

// DONE AS
CSRMatrix* _raw_kernel_2_as(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;

    // row_ptr for A (n+1), we'll fill progressively
    int *row_ptr = (int*)calloc(n + 1, sizeof(int));
    if (!row_ptr) { perror("calloc row_ptr"); exit(1); }

    // initial capacity for col_ind/val; will grow as needed
    int capacity = (B->nnz > 0 ? B->nnz : 1); // heuristic lower bound
    int *col_ind = (int*)malloc(capacity * sizeof(int));
    double *val = (double*)malloc(capacity * sizeof(double));
    if (!col_ind || !val) { perror("malloc col_ind/val"); exit(1); }

    // SPA temporary arrays:
    // accum: dense accumulator of length m (columns), reused per row
    // marker: int array length m, initialized to -1; marker[j] != -1 means j in cols_seen
    double *accum = (double*)calloc(m, sizeof(double));
    int *marker = (int*)malloc(m * sizeof(int));
    if (!accum || !marker) { perror("SPA alloc"); exit(1); }
    for (int j = 0; j < m; j++) marker[j] = -1;

    // buffer to hold columns seen in current row
    int *cols_seen = (int*)malloc(m * sizeof(int));
    if (!cols_seen) { perror("cols_seen malloc"); exit(1); }

    int nnz = 0;
    row_ptr[0] = 0;

    for (int i = 0; i < n; i++) {
        int seen_count = 0;

        // UN loop: iterate k sparsely over nonzeros in row i of B
        for (int pB = B->row_ptr[i]; pB < B->row_ptr[i + 1]; pB++) {
            int k = B->col_ind[pB];
            double bval = B->val[pB];

            // iterate nonzeros in row k of C (sparse)
            for (int pC = C->row_ptr[k]; pC < C->row_ptr[k + 1]; pC++) {
                int j = C->col_ind[pC];
                double cval = C->val[pC];
                double prod = bval * cval;

                if (marker[j] == -1) {
                    // first time we touch column j for this row i
                    marker[j] = seen_count;
                    cols_seen[seen_count] = j;
                    accum[j] = prod;
                    seen_count++;
                } else {
                    // already seen, accumulate
                    accum[j] += prod;
                }
            }
        }

        if (seen_count > 0) {
            // sort cols_seen[0..seen_count-1] so CSR has sorted column indices
            qsort(cols_seen, seen_count, sizeof(int), int_cmp);

            // append sorted entries to col_ind/val, resizing if necessary
            for (int t = 0; t < seen_count; t++) {
                int j = cols_seen[t];
                double v = accum[j];
                // optional tiny threshold to drop numerical noise
                if (fabs(v) <= 1e-15) {
                    // treat as zero; clean up marker & accum and continue
                    accum[j] = 0.0;
                    marker[j] = -1;
                    continue;
                }

                if (nnz >= capacity) {
                    capacity <<= 1;
                    if (capacity < 1) capacity = 1;
                    col_ind = (int*)realloc(col_ind, capacity * sizeof(int));
                    val = (double*)realloc(val, capacity * sizeof(double));
                    if (!col_ind || !val) { perror("realloc"); exit(1); }
                }
                col_ind[nnz] = j;
                val[nnz] = v;
                nnz++;

                // cleanup for next row
                accum[j] = 0.0;
                marker[j] = -1;
            }
        }

        row_ptr[i + 1] = nnz;
    }

    // free SPA temporaries
    free(accum);
    free(marker);
    free(cols_seen);

    // shrink-to-fit col_ind/val
    if (nnz < capacity) {
        col_ind = (int*)realloc(col_ind, nnz * sizeof(int));
        val = (double*)realloc(val, nnz * sizeof(double));
        // it's okay if realloc returns NULL when nnz==0 for some libc; we skip check
    }

    CSRMatrix *A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    if (!A) { perror("malloc A"); exit(1); }
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}


// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j)
// DONE NS
static CSRMatrix* _raw_kernel_2_ns(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;

    // Prepare CSR data (dynamic arrays)
    int *row_ptr = (int *)malloc((n + 1) * sizeof(int));
    int *col_ind = NULL;
    double *val = NULL;
    int nnz = 0;

    // Sparse accumulator (SPA)
    double *spa = (double *)calloc(m, sizeof(double));
    int *marker = (int *)malloc(m * sizeof(int));
    for (int j = 0; j < m; j++) marker[j] = -1;

    row_ptr[0] = 0;

    // Iterate over rows of B
    for (int i = 0; i < n; i++) {
        int row_nnz = 0;

        // Iterate over columns of B (reduction axis k)
        for (int k = 0; k < B->cols; k++) {
            double b_val = B->get_element(i, k);
            if (b_val == 0.0) continue;

            // For every j (explicitly)
            for (int j = 0; j < C->cols; j++) {
                double c_val = C->get_element(k, j);
                if (c_val == 0.0) continue;

                if (marker[j] != i) {  // not yet touched in this row
                    marker[j] = i;
                    spa[j] = b_val * c_val;
                } else {
                    spa[j] += b_val * c_val;
                }
            }
        }

        // Compact the SPA to CSR row
        for (int j = 0; j < m; j++) {
            if (marker[j] == i && spa[j] != 0.0) {
                nnz++;
                col_ind = (int *)realloc(col_ind, nnz * sizeof(int));
                val = (double *)realloc(val, nnz * sizeof(double));
                col_ind[nnz - 1] = j;
                val[nnz - 1] = spa[j];
                spa[j] = 0.0;  // reset
            }
        }

        row_ptr[i + 1] = nnz;
    }

    free(spa);
    free(marker);

    CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j)
static CSRMatrix* _raw_kernel_2_af(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;
    int p = B->cols; // reduction dim

    // Pre-allocate space; we'll resize later if needed
    int capacity = n * 8;  // initial guess
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val   = (double *)malloc(capacity * sizeof(double));

    int nnz = 0;
    row_ptr[0] = 0;

    // Sparse accumulator (SPA) per row
    double *accum = (double *)calloc(m, sizeof(double));
    int *marker = (int *)malloc(m * sizeof(int));
    for (int j = 0; j < m; j++) marker[j] = -1;

    for (int i = 0; i < n; i++) {
        std::vector<int> cols_seen;

        // iterate through dense k
        for (int k = 0; k < p; k++) {
            double b_val = B->get_element(i, k);
            if (b_val == 0.0) continue;

            // iterate sparse over row k of C
            for (int pC = C->row_ptr[k]; pC < C->row_ptr[k + 1]; pC++) {
                int j = C->col_ind[pC];
                double c_val = C->val[pC];

                if (marker[j] != i) {
                    marker[j] = i;
                    accum[j] = b_val * c_val;
                    cols_seen.push_back(j);
                } else {
                    accum[j] += b_val * c_val;
                }
            }
        }

        // flush SPA into CSR arrays
        for (int idx : cols_seen) {
            if (nnz >= capacity) {
                capacity *= 2;
                col_ind = (int *)realloc(col_ind, capacity * sizeof(int));
                val     = (double *)realloc(val, capacity * sizeof(double));
            }
            col_ind[nnz] = idx;
            val[nnz] = accum[idx];
            nnz++;
            accum[idx] = 0.0; // reset
        }

        row_ptr[i + 1] = nnz;
    }

    free(accum);
    free(marker);

    CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j)
// DONE RS
static CSRMatrix* _raw_kernel_2_rs(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;

    // Prepare CSR data (dynamic arrays)
    int *row_ptr = (int *)malloc((n + 1) * sizeof(int));
    int *col_ind = NULL;
    double *val = NULL;
    int nnz = 0;

    // Sparse accumulator (SPA)
    double *spa = (double *)calloc(m, sizeof(double));
    int *marker = (int *)malloc(m * sizeof(int));
    for (int j = 0; j < m; j++) marker[j] = -1;

    row_ptr[0] = 0;

    // Iterate over rows of B
    for (int i = 0; i < n; i++) {
        int row_nnz = 0;

        // Iterate over columns of B (reduction axis k)
        for (int ki = B->row_ptr[i]; ki < B->row_ptr[i+1]; ki++)
        {
            double b_val = B->val[ki];
            int k = B->col_ind[ki];
            for (int j = 0; j < C->cols; j++)
            {
                double c_val = C->get_element(k, j);
                if (c_val == 0.0) continue;

                if (marker[j] != i)
                {
                    marker[j] = i;
                    spa[j] = b_val * c_val;
                } else {
                    spa[j] += b_val * c_val;
                }
            }
        }

        // Compact the SPA to CSR row
        for (int j = 0; j < m; j++) {
            if (marker[j] == i && spa[j] != 0.0) {
                nnz++;
                col_ind = (int *)realloc(col_ind, nnz * sizeof(int));
                val = (double *)realloc(val, nnz * sizeof(double));
                col_ind[nnz - 1] = j;
                val[nnz - 1] = spa[j];
                spa[j] = 0.0;  // reset
            }
        }

        row_ptr[i + 1] = nnz;
    }

    free(spa);
    free(marker);

    CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j) * D(j,k)
// DONE
static CSRMatrix* _raw_kernel_2_1ns(const CSRMatrix *B, const CSRMatrix *C, const CSRMatrix *D) {
    if (B->cols != C->rows || C->rows != D->rows || C->cols != D->cols) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d), C(%d,%d), D(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols, D->rows, D->cols);
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;

    int *row_ptr = (int *)malloc((n + 1) * sizeof(int));
    int *col_ind = NULL;
    double *val = NULL;
    int nnz = 0;

    // Sparse accumulator (SPA)
    double *spa = (double *)calloc(m, sizeof(double));
    int *marker = (int *)malloc(m * sizeof(int));
    for (int j = 0; j < m; j++) marker[j] = -1;

    row_ptr[0] = 0;

    // Iterate over rows of B
    for (int i = 0; i < n; i++) {
        // Iterate over nonzeros in B[i, :]
        // Iterate over columns of B
        for (int k = 0; k < B->cols; k++) {
            double b_val = B->get_element(i, k);
            if (b_val == 0.0) continue;

            // For every j (explicitly)
            for (int j = 0; j < C->cols; j++) {
                double c_val = C->get_element(k, j);
                if (c_val == 0.0) continue;

                double d_val = D->get_element(j, k);
                if (d_val == 0.0) continue;

                double prod = b_val * c_val * d_val;

                if (marker[j] != i) {
                    marker[j] = i;
                    spa[j] = prod;
                } else {
                    spa[j] += prod;
                }
            }
        }

        // Compact SPA into CSR row
        for (int j = 0; j < m; j++) {
            if (marker[j] == i && spa[j] != 0.0) {
                nnz++;
                col_ind = (int *)realloc(col_ind, nnz * sizeof(int));
                val = (double *)realloc(val, nnz * sizeof(double));
                col_ind[nnz - 1] = j;
                val[nnz - 1] = spa[j];
                spa[j] = 0.0;
            }
        }

        row_ptr[i + 1] = nnz;
    }

    free(spa);
    free(marker);

    CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j) * D(j,k)
// DONE
static CSRMatrix* _raw_kernel_2_1as(const CSRMatrix *B, const CSRMatrix *C, const CSRMatrix *D) {
    if (B->cols != C->rows || C->rows != D->rows || C->cols != D->cols) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d), C(%d,%d), D(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols, D->rows, D->cols);
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;

    int *row_ptr = (int *)malloc((n + 1) * sizeof(int));
    int *col_ind = NULL;
    double *val = NULL;
    int nnz = 0;

    // Sparse accumulator (SPA)
    double *spa = (double *)calloc(m, sizeof(double));
    int *marker = (int *)malloc(m * sizeof(int));
    for (int j = 0; j < m; j++) marker[j] = -1;

    row_ptr[0] = 0;

    // Iterate over rows of B
    for (int i = 0; i < n; i++) {
        // Iterate over nonzeros in B[i, :]
        for (int idx_B = B->row_ptr[i]; idx_B < B->row_ptr[i+1]; idx_B++) {
            int k = B->col_ind[idx_B];
            double b_val = B->val[idx_B];
            if (b_val == 0.0) continue;

            // For every j (explicitly)
            for (int idx_C = C->row_ptr[k]; idx_C < C->row_ptr[k+1]; idx_C++)
            {
                int j = C->col_ind[idx_C];
                double c_val = C->val[idx_C];
                double temp_product = 0.0;
                if (c_val == 0.0) continue;

                for (int idx_D = D->row_ptr[j]; idx_D < D->row_ptr[j+1]; idx_D++)
                {
                    double d_val = D->val[idx_D];
                    if (d_val == 0.0) continue;

                    if (D->col_ind[idx_D] != k) continue;
                    temp_product = c_val * d_val;
                    break;
                }

                if (temp_product != 0.0)
                {
                    if (marker[j] != i) {
                        marker[j] = i;
                        spa[j] = (b_val * temp_product);
                    } else {
                        spa[j] += (b_val * temp_product);
                    }
                }
            }
        }

        // Compact SPA into CSR row
        for (int j = 0; j < m; j++) {
            if (marker[j] == i && spa[j] != 0.0) {
                nnz++;
                col_ind = (int *)realloc(col_ind, nnz * sizeof(int));
                val = (double *)realloc(val, nnz * sizeof(double));
                col_ind[nnz - 1] = j;
                val[nnz - 1] = spa[j];
                spa[j] = 0.0;
            }
        }

        row_ptr[i + 1] = nnz;
    }

    free(spa);
    free(marker);

    CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j) * D(j,k)
// DONE
static CSRMatrix* _raw_kernel_2_1rs(const CSRMatrix *B, const CSRMatrix *C, const CSRMatrix *D) {
    if (B->cols != C->rows || C->rows != D->rows || C->cols != D->cols) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d), C(%d,%d), D(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols, D->rows, D->cols);
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;

    int *row_ptr = (int *)malloc((n + 1) * sizeof(int));
    int *col_ind = NULL;
    double *val = NULL;
    int nnz = 0;

    // Sparse accumulator (SPA)
    double *spa = (double *)calloc(m, sizeof(double));
    int *marker = (int *)malloc(m * sizeof(int));
    for (int j = 0; j < m; j++) marker[j] = -1;

    row_ptr[0] = 0;

    // Iterate over rows of B
    for (int i = 0; i < n; i++) {
        // Iterate over nonzeros in B[i, :]
        for (int idx_B = B->row_ptr[i]; idx_B < B->row_ptr[i+1]; idx_B++) {
            int k = B->col_ind[idx_B];
            double b_val = B->val[idx_B];

            // For every j (explicitly)
            for (int j = 0; j < C->cols; j++) {
                double c_val = C->get_element(k, j);
                if (c_val == 0.0) continue;

                double d_val = D->get_element(j, k);
                if (d_val == 0.0) continue;

                double prod = b_val * c_val * d_val;

                if (marker[j] != i) {
                    marker[j] = i;
                    spa[j] = prod;
                } else {
                    spa[j] += prod;
                }
            }
        }

        // Compact SPA into CSR row
        for (int j = 0; j < m; j++) {
            if (marker[j] == i && spa[j] != 0.0) {
                nnz++;
                col_ind = (int *)realloc(col_ind, nnz * sizeof(int));
                val = (double *)realloc(val, nnz * sizeof(double));
                col_ind[nnz - 1] = j;
                val[nnz - 1] = spa[j];
                spa[j] = 0.0;
            }
        }

        row_ptr[i + 1] = nnz;
    }

    free(spa);
    free(marker);

    CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j) * D(j,k)
static CSRMatrix* _raw_kernel_2_1af(const CSRMatrix *B, const CSRMatrix *C, const CSRMatrix *D) {
    if (B->cols != C->rows || C->rows != D->rows || C->cols != D->cols) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d), C(%d,%d), D(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols, D->rows, D->cols);
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;

    int *row_ptr = (int *)malloc((n + 1) * sizeof(int));
    int *col_ind = NULL;
    double *val = NULL;
    int nnz = 0;

    // Sparse accumulator (SPA)
    double *spa = (double *)calloc(m, sizeof(double));
    int *marker = (int *)malloc(m * sizeof(int));
    for (int j = 0; j < m; j++) marker[j] = -1;

    row_ptr[0] = 0;

    // Iterate over rows of B
    for (int i = 0; i < n; i++) {
        // Iterate over nonzeros in B[i, :]
        for (int idx_B = B->row_ptr[i]; idx_B < B->row_ptr[i+1]; idx_B++) {
            int k = B->col_ind[idx_B];
            double b_val = B->val[idx_B];
            if (b_val == 0.0) continue;

            // For every j (explicitly)
            for (int idx_C = C->row_ptr[k]; idx_C < C->row_ptr[k+1]; idx_C++)
            {
                int j = C->col_ind[idx_C];
                double c_val = C->val[idx_C];
                if (c_val == 0.0) continue;

                double d_val = D->get_element(j,k);
                if (d_val == 0.0) continue;

                double prod = b_val * c_val * d_val;
                
                if (marker[j] != i) {
                    marker[j] = i;
                    spa[j] = prod;
                } else {
                    spa[j] += prod;
                }
            }
        }

        // Compact SPA into CSR row
        for (int j = 0; j < m; j++) {
            if (marker[j] == i && spa[j] != 0.0) {
                nnz++;
                col_ind = (int *)realloc(col_ind, nnz * sizeof(int));
                val = (double *)realloc(val, nnz * sizeof(double));
                col_ind[nnz - 1] = j;
                val[nnz - 1] = spa[j];
                spa[j] = 0.0;
            }
        }

        row_ptr[i + 1] = nnz;
    }

    free(spa);
    free(marker);

    CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j)
// DONE
static CSRMatrix* _raw_kernel_2_1(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = B->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = n * m;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));
    
    int nnz = 0;
    row_ptr[0] = 0;

    // iterate over the row of B
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            double accum = 0.0;
            for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++)
            {
                int j = B->col_ind[p];
                // search on C
                for (int q = C->row_ptr[j]; q < C->row_ptr[j+1]; q++)
                {
                    if (C->col_ind[q] == k)
                    {
                        accum += (B->val[p] * C->val[q]);
                        break;
                    }
                }
            }
            if (accum != 0.0)
            {
                col_ind[nnz] = k;
                val[nnz] = accum;
                nnz++;
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;
    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j)
static CSRMatrix* _raw_kernel_2_2(const CSRMatrix *B, const CSCMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = B->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = n * m;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));
    
    int nnz = 0;
    row_ptr[0] = 0;

    // iterate over the row of B
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < m; k++)
        {
            double accum = 0.0;
            for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++)
            {
                int j = B->col_ind[p];

                // search on C
                for (int q = C->col_ptr[k]; q < C->col_ptr[k+1]; q++)
                {
                    if (C->row_ind[q] == B->col_ind[p])
                    {
                        accum += (B->val[p] * C->val[q]);
                        break;
                    }
                }
            }
            if (accum != 0.0)
            {
                col_ind[nnz] = k;
                val[nnz] = accum;
                nnz++;
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j)
static CSRMatrix* _raw_kernel_2_3(const CSRMatrix *B, const COOMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = B->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = n * m;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));
    
    int nnz = 0;
    row_ptr[0] = 0;

    // iterate over the row of B
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < m; k++)
        {
            double accum = 0.0;
            for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++)
            {
                int j = B->col_ind[p];

                // search on C
                for (int q = 0; q < C->values.size(); q++)
                {
                    if (C->col_indices[q] == k && C->row_indices[q] == j)
                    {
                        accum += (B->val[p] * C->values[q]);
                        break; 
                    }
                }
            }
            if (accum != 0.0)
            {
                col_ind[nnz] = k;
                val[nnz] = accum;
                nnz++;
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}


// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j)
static CSRMatrix* _raw_kernel_2_4(const CSCMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = B->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = n * m;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));
    
    int nnz = 0;
    row_ptr[0] = 0;

    // iterate over the row of B
    for (int i = 0; i < n; i++)
    {
        for (int kk = 0; kk < m; kk++) 
        {
            double accum = 0.0;
            for (int j = 0; j < m; j++)
            {
                for (int q = B->col_ptr[j]; q < B->col_ptr[j+1]; q++)
                {
                    if (B->row_ind[q] == i)
                    {
                        // search on C
                        for (int k = C->row_ptr[j]; k < C->row_ptr[j+1]; k++)
                        {
                            if (C->col_ind[k] == kk)
                            {
                                accum += (B->val[q] * C->val[k]);
                                break;
                            }
                        }
                    }
                }
            }
            if (accum != 0.0)
            {
                col_ind[nnz] = kk;
                val[nnz] = accum;
                nnz++;
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j)
static CSRMatrix* _raw_kernel_2_5(const CSCMatrix *B, const CSCMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = B->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = n * m;
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));
    
    int nnz = 0;
    row_ptr[0] = 0;

    // iterate over the row of B
    for (int i = 0; i < n; i++)
    {
        for (int r = 0; r < m; r++)
        {
            double accum = 0.0;
            for (int j = 0; j < m; j++)
            {
                for (int p = B->col_ptr[j]; p < B->col_ptr[j+1]; p++)
                {
                    if (B->row_ind[p] == i)
                    {
                        // search on C
                        for (int k = C->col_ptr[r]; k < C->col_ptr[r+1]; k++)
                        {
                            if (C->row_ind[k] == j)
                            {
                                accum += (B->val[p] * C->val[k]);
                                break;
                            }
                        }
                    }
                }
            }
            if (accum != 0.0)
            {
                col_ind[nnz] = r;
                val[nnz] = accum;
                nnz++;
            }
        }
        row_ptr[i+1] = nnz;
    }
    row_ptr[n] = nnz;

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}


// A(i, j) = B(i, k)  * C(k, j) * D(k, j)
static CSRMatrix* _raw_kernel_3_1(const CSRMatrix *B, const CSRMatrix *C, const CSRMatrix *D) {
    if (B->cols != C->rows || C->rows != D->rows || C->cols != D->cols) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d), C(%d,%d), D(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols, D->rows, D->cols);
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = B->nnz * 4; // heuristic
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));
    int nnz = 0;
    // iterate through the rows of B
    for (int i = 0; i < n; i++)
    {
        // get individual values on the row i of B
        for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++)
        {
            int k = B->col_ind[p];
            
        }
    }

    CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;

    // A->print_dense();
    return A;
}

// A(i) = B(i, j) * C(j, i)
static CSRMatrix* _raw_kernel_4_1(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->cols) { // B(i,j) * C(j,i) => check B.cols == C.cols
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = 1; // A is 1D of size n
    int *row_ptr = (int*)calloc(n+1, sizeof(int));
    int capacity = B->nnz; // heuristic
    int *col_ind = (int*)malloc(capacity * sizeof(int));
    double *val = (double*)malloc(capacity * sizeof(double));

    double *accum = (double*)calloc(n, sizeof(double)); // accumulator for A(i)
    
    int nnz = 0;
    row_ptr[0] = 0;

    for (int i = 0; i < n; i++) {
        double sum = 0.0;

        // iterate over non-zeros of row i in B
        for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++) {
            int j = B->col_ind[p];
            double bval = B->val[p];

            // access C(j,i)
            // int q = C->row_ptr[j];
            // while (q < C->row_ptr[j+1])
            // {
            //     if (C->col_ind[q] == i)
            //     {
            //         sum += bval * C->val[q];
            //         break;
            //     }
            //     q++;
            // }
            for (int q = C->row_ptr[j]; q < C->row_ptr[j+1]; q++) {
                // if (C->col_ind[q]>i) break;
                if (C->col_ind[q] == i) {
                    sum += bval * C->val[q];
                    break; // only one (j,i) entry assumed
                }
            }
        }

        if (fabs(sum) > 1e-12) { // ignore zeros
            if (nnz >= capacity) {
                capacity *= 2;
                col_ind = (int*)realloc(col_ind, capacity * sizeof(int));
                val = (double*)realloc(val, capacity * sizeof(double));
            }
            col_ind[nnz] = 0; // single column
            val[nnz] = sum;
            nnz++;
        }

        row_ptr[i+1] = nnz;
    }

    free(accum);

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// A(i) = B(i, j, k) * C(i, k, j)
static CSRMatrix* _raw_kernel_5_1(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->cols) { // B(i,j) * C(j,i) => check B.cols == C.cols
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = 1; // A is 1D of size n
    int *row_ptr = (int*)calloc(n+1, sizeof(int));
    int capacity = B->nnz; // heuristic
    int *col_ind = (int*)malloc(capacity * sizeof(int));
    double *val = (double*)malloc(capacity * sizeof(double));

    double *accum = (double*)calloc(n, sizeof(double)); // accumulator for A(i)
    
    int nnz = 0;
    row_ptr[0] = 0;

    for (int i = 0; i < n; i++) {
        double sum = 0.0;

        // iterate over non-zeros of row i in B
        for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++) {
            int j = B->col_ind[p];
            double bval = B->val[p];

            // access C(j,i)
            // int q = C->row_ptr[j];
            // while (q < C->row_ptr[j+1])
            // {
            //     if (C->col_ind[q] == i)
            //     {
            //         sum += bval * C->val[q];
            //         break;
            //     }
            //     q++;
            // }
            for (int q = C->row_ptr[j]; q < C->row_ptr[j+1]; q++) {
                // if (C->col_ind[q]>i) break;
                if (C->col_ind[q] == i) {
                    sum += bval * C->val[q];
                    break; // only one (j,i) entry assumed
                }
            }
        }

        if (fabs(sum) > 1e-12) { // ignore zeros
            if (nnz >= capacity) {
                capacity *= 2;
                col_ind = (int*)realloc(col_ind, capacity * sizeof(int));
                val = (double*)realloc(val, capacity * sizeof(double));
            }
            col_ind[nnz] = 0; // single column
            val[nnz] = sum;
            nnz++;
        }

        row_ptr[i+1] = nnz;
    }

    free(accum);

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

// A(i) = B(i, j, k) * C(i, k, j)
double raw_kernel_5_1(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_5_1(&csrB, &csrC);
    // std::cout << "=============== Unzipper Result ===============" << std::endl;
    // result->print();
    // std::cout << "===============================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

// A(i) = B(i, j) * C(j, i)
double raw_kernel_4_1(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_4_1(&csrB, &csrC);
    // std::cout << "=============== Unzipper Result ===============" << std::endl;
    // result->print();
    // std::cout << "===============================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

// A(i, j) = B(i, k)  * C(k, j) * D(k, j)
double raw_kernel_3_1(const COOMatrix &B, const COOMatrix &C, const COOMatrix &D)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC, csrD;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    COO_to_CSR(D, csrD);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_3_1(&csrB, &csrC, &csrD);
    // std::cout << "=============== Unzipper Result ===============" << std::endl;
    // result->print();
    // std::cout << "===============================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(&csrD);
    // freeCSRMatrix(result);
    // free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
double raw_kernel_1_1(const COOMatrix &B, const COOMatrix &C) 
{
    struct timespec start, end;
    CSRMatrix csrB, csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_1_1(&csrB, &csrC);
    std::cout << "============= CSR CSR CSR =============" << std::endl;
    result->print();
    std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_2(const COOMatrix &B, const COOMatrix &C) 
{
    struct timespec start, end;
    CSRMatrix csrB;
    CSCMatrix cscC;
    COO_to_CSR(B, csrB);
    COO_to_CSC(C, cscC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_1_2(&csrB, &cscC);
    // std::cout << "============= CSR CSR CSC =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSCMatrix(&cscC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_3(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB;
    COO_to_CSR(B, csrB);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_1_3(&csrB, &C);
    // std::cout << "============= CSR CSR COO =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_7(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrC;
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_1_7(&B, &csrC);
    // std::cout << "============= CSR COO CSR =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_8(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSCMatrix cscC;
    COO_to_CSC(C, cscC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_1_8(&B, &cscC);
    // std::cout << "============= CSR COO CSC =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSCMatrix(&cscC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_9(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_1_9(&B, &C);
    // std::cout << "============= CSR COO COO =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_13(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSCMatrix cscb;
    CSRMatrix csrc;
    COO_to_CSC(B, cscb);
    COO_to_CSR(C, csrc);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSCMatrix* result = _raw_kernel_1_13(&cscb, &csrc);
    // std::cout << "============= CSC CSC CSR =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrc);
    freeCSCMatrix(&cscb);
    freeCSCMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_14(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSCMatrix cscB;
    CSCMatrix cscC;
    COO_to_CSC(B, cscB);
    COO_to_CSC(C, cscC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSCMatrix* result = _raw_kernel_1_14(&cscB, &cscC);
    // std::cout << "============= CSC CSC CSR =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSCMatrix(&cscC);
    freeCSCMatrix(&cscB);
    freeCSCMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_15(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSCMatrix cscB;
    COO_to_CSC(B, cscB);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSCMatrix* result = _raw_kernel_1_15(&cscB, &C);
    // std::cout << "============= CSC CSC COO =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSCMatrix(&cscB);
    freeCSCMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_24(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB;
    CSRMatrix csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSCMatrix* result = _raw_kernel_1_24(&csrB, &csrC);
    // std::cout << "============= CSC CSR CSR =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSCMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_25(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB;
    CSCMatrix cscC;
    COO_to_CSR(B, csrB);
    COO_to_CSC(C, cscC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSCMatrix* result = _raw_kernel_1_25(&csrB, &cscC);
    // std::cout << "============= CSC CSR CSC =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSCMatrix(&cscC);
    freeCSCMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_26(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB;
    COO_to_CSR(B, csrB);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSCMatrix* result = _raw_kernel_1_26(&csrB, &C);
    // std::cout << "============= CSC CSR COO =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSCMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_33(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB;
    CSRMatrix csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_1_33(&csrB, &csrC);
    // std::cout << "============= CSR CSR CSR =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_34(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB;
    CSRMatrix csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_1_34(&csrB, &csrC);
    // std::cout << "============= CSR CSR CSR =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_1_35(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB;
    CSCMatrix cscC;
    COO_to_CSR(B, csrB);
    COO_to_CSC(C, cscC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_1_35(&csrB, &cscC);
    // std::cout << "============= CSR CSR CSC =============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSCMatrix(&cscC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

// Multiply A = B * C (MatMul)
// A(i, j) = B(i, k) * C(k, j)
double raw_kernel_2_1(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_1(&csrB, &csrC);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSR CSR ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_2_ns(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_ns(&csrB, &csrC);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSR CSR NS ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_2_as(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_as(&csrB, &csrC);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSR CSR AS ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_2_rs(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_rs(&csrB, &csrC);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSR CSR RS ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_2_af(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_af(&csrB, &csrC);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSR CSR AF ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_2_1ns(const COOMatrix &B, const COOMatrix &C, const COOMatrix &D)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC, csrD;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    COO_to_CSR(D, csrD);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_1ns(&csrB, &csrC, &csrD);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSR CSR CSR NS ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(&csrD);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_2_1as(const COOMatrix &B, const COOMatrix &C, const COOMatrix &D)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC, csrD;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    COO_to_CSR(D, csrD);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_1as(&csrB, &csrC, &csrD);
    clock_gettime(CLOCK_MONOTONIC, &end);
    std::cout << "============ CSR CSR CSR CSR AS ==============" << std::endl;
    result->print();
    std::cout << "=======================================" << std::endl;
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(&csrD);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_2_1rs(const COOMatrix &B, const COOMatrix &C, const COOMatrix &D)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC, csrD;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    COO_to_CSR(D, csrD);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_1rs(&csrB, &csrC, &csrD);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSR CSR CSR RS ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(&csrD);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_2_1af(const COOMatrix &B, const COOMatrix &C, const COOMatrix &D)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC, csrD;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    COO_to_CSR(D, csrD);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_1af(&csrB, &csrC, &csrD);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSR CSR CSR AF ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(&csrD);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}


double raw_kernel_2_2(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB;
    CSCMatrix cscC;
    COO_to_CSR(B, csrB);
    COO_to_CSC(C, cscC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_2(&csrB, &cscC);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSR CSC ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSRMatrix(&csrB);
    freeCSCMatrix(&cscC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

double raw_kernel_2_3(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrB;
    COO_to_CSR(B, csrB);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_3(&csrB, &C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSR COO ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSRMatrix(&csrB);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

// Multiply A = B * C (MatMul)
// A(i, j) = B(i, k) * C(k, j)
double raw_kernel_2_4(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSRMatrix csrC;
    CSCMatrix cscB;
    COO_to_CSC(B, cscB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_4(&cscB, &csrC);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSC CSR ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSCMatrix(&cscB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

// Multiply A = B * C (MatMul)
// A(i, j) = B(i, k) * C(k, j)
double raw_kernel_2_5(const COOMatrix &B, const COOMatrix &C)
{
    struct timespec start, end;
    CSCMatrix cscB, cscC;
    COO_to_CSC(B, cscB);
    COO_to_CSC(C, cscC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_2_5(&cscB, &cscC);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // std::cout << "============ CSR CSC CSC ==============" << std::endl;
    // result->print();
    // std::cout << "=======================================" << std::endl;
    freeCSCMatrix(&cscB);
    freeCSCMatrix(&cscC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}