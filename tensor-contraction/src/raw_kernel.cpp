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

// Multiply A = B * C (Mat-Mul)
// A(i, j) = B(i, k) * C(k, j)
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
                    if (C->row_ind[q] == j)
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


// // A(i, j) = B(i, k)  * C(k, j) * D(k, j)
// static CSRMatrix* _raw_kernel_3_1(const CSRMatrix *B, const CSRMatrix *C, const CSRMatrix *D) {
//     if (B->cols != C->rows || C->rows != D->rows || C->cols != D->cols) {
//         fprintf(stderr, "Dimension mismatch: B(%d,%d), C(%d,%d), D(%d,%d)\n",
//                 B->rows, B->cols, C->rows, C->cols, D->rows, D->cols);
//         exit(1);
//     }

//     int n = B->rows;
//     int m = C->cols;
//     int *row_ptr = (int *)calloc(n + 1, sizeof(int));
//     int capacity = B->nnz * 4; // heuristic
//     int *col_ind = (int *)malloc(capacity * sizeof(int));
//     double *val = (double *)malloc(capacity * sizeof(double));

//     // Temporary buffers
//     double *accum = (double *)calloc(m, sizeof(double));
//     char *flag = (char *)calloc(m, sizeof(char));
//     int *cols_in_rows = (int *)malloc(m * sizeof(int));

//     int nnz = 0;
//     row_ptr[0] = 0;

//     for (int i = 0; i < n; i++) {
//         int count = 0;

//         // iterate over nonzeros of B[i,*]
//         for (int p = B->row_ptr[i]; p < B->row_ptr[i + 1]; p++) {
//             int k = B->col_ind[p];
//             double bval = B->val[p];

//             // merge multiply row k of C and D
//             int qc = C->row_ptr[k];
//             int qd = D->row_ptr[k];

//             // both C and D are sparse; merge step
//             while (qc < C->row_ptr[k + 1] && qd < D->row_ptr[k + 1]) {
//                 int jc = C->col_ind[qc];
//                 int jd = D->col_ind[qd];

//                 if (jc == jd) {
//                     int j = jc;
//                     double cval = C->val[qc];
//                     double dval = D->val[qd];
//                     double prod = bval * cval * dval;

//                     if (!flag[j]) {
//                         flag[j] = 1;
//                         cols_in_rows[count++] = j;
//                         accum[j] = prod;
//                     } else {
//                         accum[j] += prod;
//                     }

//                     qc++;
//                     qd++;
//                 } else if (jc < jd) {
//                     qc++;
//                 } else {
//                     qd++;
//                 }
//             }
//         }

//         // sort columns
//         qsort(cols_in_rows, count, sizeof(int), cmp_int);

//         // output this row
//         for (int a = 0; a < count; a++) {
//             int j = cols_in_rows[a];
//             double v = accum[j];
//             if (v != 0.0) {
//                 if (nnz >= capacity) {
//                     capacity *= 2;
//                     col_ind = (int *)realloc(col_ind, capacity * sizeof(int));
//                     val = (double *)realloc(val, capacity * sizeof(double));
//                 }
//                 col_ind[nnz] = j;
//                 val[nnz] = v;
//                 nnz++;
//             }
//             flag[j] = 0;
//             accum[j] = 0.0;
//         }

//         row_ptr[i + 1] = nnz;
//     }

//     free(accum);
//     free(flag);
//     free(cols_in_rows);

//     CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
//     A->rows = n;
//     A->cols = m;
//     A->nnz = nnz;
//     A->row_ptr = row_ptr;
//     A->col_ind = col_ind;
//     A->val = val;

//     // A->print_dense();
//     return A;
// }

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
    // CSRMatrix* result = _raw_kernel_3_1(&csrB, &csrC, &csrD);
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