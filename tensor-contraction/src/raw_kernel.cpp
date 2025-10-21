#include "raw_kernel.hpp"


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
                    // break;
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

int cmp_int(const void *a, const void *b) {
    int x = *(int*)a;
    int y = *(int*)b;
    return (x > y) - (x < y);  // returns 1, 0, or -1
}

static CSRMatrix* _raw_kernel_2_1(const CSRMatrix *B, const CSRMatrix *C) {
    if (B->cols != C->rows) {
        fprintf(stderr, "Dimension mismatch: B(%d,%d) * C(%d,%d)\n",
                B->rows, B->cols, C->rows, C->cols);
        exit(1);
    }

    int n = B->rows;
    int m = C->cols;
    int *row_ptr = (int *)calloc(n + 1, sizeof(int));
    int capacity = B->nnz * 4; // heuristic
    int *col_ind = (int *)malloc(capacity * sizeof(int));
    double *val = (double *)malloc(capacity * sizeof(double));
    
    // Temporary arrays for onw row
    double *accum = (double *)calloc(m, sizeof(double));
    char *flag = (char *)calloc(m, sizeof(char));
    int *cols_in_rows = (int *)malloc(m* sizeof(int));
    
    int nnz = 0;
    row_ptr[0] = 0;

    // iterate over the row of B
    for (int i = 0; i < n; i++) {
        int count =0;

        // iterate over the non-zeros of i-th row of B
        for (int p = B->row_ptr[i]; p < B->row_ptr[i+1]; p++) {
            int k = B->col_ind[p];
            double bval = B->val[p];

            // Multiply with row k of C
            int q = C->row_ptr[k];
            while (q < C->row_ptr[k+1])
            {
                int j = C->col_ind[q];
                if (!flag[j])
                {
                    flag[j] = 1;
                    cols_in_rows[count++] = j;
                    // double cval = C->val[q];
                    accum[j] = bval * (C->val[q]);
                } else {
                    accum[j] += (bval * (C->val[q]));
                }

                q++;
            }
            // for (int q = C->row_ptr[k]; q < C->row_ptr[k+1]; q++)
            // {
            //     int j = C->col_ind[q];
            //     double cval = C->val[q];
            //     // double res = bval * cval;

            //     if (!flag[j])
            //     {
            //         flag[j] = 1;
            //         cols_in_rows[count++] = j;
            //         accum[j] = bval * cval;
            //     } else {
            //         accum[j] += bval * cval;
            //     }
            // }
        }

        // Sort columns (insertion sort)
        // for (int a = 1; a < count; a++)
        // {
        //     int cj = cols_in_rows[a];
        //     int b = a - 1;
        //     while (b >= 0 && cols_in_rows[b] > cj)
        //     {
        //         cols_in_rows[b + 1] = cols_in_rows[b];
        //         b--;
        //     }
        //     cols_in_rows[b+1] = cj;
        // }
        qsort(cols_in_rows, count, sizeof(int), cmp_int);

        // output row
        for (int a = 0; a < count; a++)
        {
            int j = cols_in_rows[a];
            if (accum[j] != 0.0)
            {
                if (nnz >= capacity)
                {
                    capacity *= 2;
                    col_ind = (int *)realloc(col_ind, capacity * sizeof(int));
                    val = (double *)realloc(val, capacity * sizeof(double));
                }
                col_ind[nnz] = j;
                val[nnz] = accum[j];
                nnz++;
            }
            flag[j] = 0;
        }
        row_ptr[i+1] = nnz;
    }
    
    free(accum);
    free(flag);
    free(cols_in_rows);

    CSRMatrix* A = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;
    return A;
}

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

    // Temporary buffers
    double *accum = (double *)calloc(m, sizeof(double));
    char *flag = (char *)calloc(m, sizeof(char));
    int *cols_in_rows = (int *)malloc(m * sizeof(int));

    int nnz = 0;
    row_ptr[0] = 0;

    for (int i = 0; i < n; i++) {
        int count = 0;

        // iterate over nonzeros of B[i,*]
        for (int p = B->row_ptr[i]; p < B->row_ptr[i + 1]; p++) {
            int k = B->col_ind[p];
            double bval = B->val[p];

            // merge multiply row k of C and D
            int qc = C->row_ptr[k];
            int qd = D->row_ptr[k];

            // both C and D are sparse; merge step
            while (qc < C->row_ptr[k + 1] && qd < D->row_ptr[k + 1]) {
                int jc = C->col_ind[qc];
                int jd = D->col_ind[qd];

                if (jc == jd) {
                    int j = jc;
                    double cval = C->val[qc];
                    double dval = D->val[qd];
                    double prod = bval * cval * dval;

                    if (!flag[j]) {
                        flag[j] = 1;
                        cols_in_rows[count++] = j;
                        accum[j] = prod;
                    } else {
                        accum[j] += prod;
                    }

                    qc++;
                    qd++;
                } else if (jc < jd) {
                    qc++;
                } else {
                    qd++;
                }
            }
        }

        // sort columns
        qsort(cols_in_rows, count, sizeof(int), cmp_int);

        // output this row
        for (int a = 0; a < count; a++) {
            int j = cols_in_rows[a];
            double v = accum[j];
            if (v != 0.0) {
                if (nnz >= capacity) {
                    capacity *= 2;
                    col_ind = (int *)realloc(col_ind, capacity * sizeof(int));
                    val = (double *)realloc(val, capacity * sizeof(double));
                }
                col_ind[nnz] = j;
                val[nnz] = v;
                nnz++;
            }
            flag[j] = 0;
            accum[j] = 0.0;
        }

        row_ptr[i + 1] = nnz;
    }

    free(accum);
    free(flag);
    free(cols_in_rows);

    CSRMatrix *A = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    A->rows = n;
    A->cols = m;
    A->nnz = nnz;
    A->row_ptr = row_ptr;
    A->col_ind = col_ind;
    A->val = val;

    return A;
}

// A(i, j) = B(i, k)  * C(k, j) * D(k, j)
CSRMatrix* raw_kernel_3_1(const COOMatrix &B, const COOMatrix &C, const COOMatrix &D)
{
    struct timespec start, end;
    CSRMatrix csrB, csrC, csrD;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    COO_to_CSR(D, csrD);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_3_1(&csrB, &csrC, &csrD);
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(&csrD);
    // freeCSRMatrix(result);
    // free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return result;
}

// Multiply A = B * C (element wise row-col multiplication)
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
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
// A = einsum("ij,ji->ij", B, C)
double raw_kernel_1_1(const COOMatrix &B, const COOMatrix &C) 
{
    struct timespec start, end;
    CSRMatrix csrB, csrC;
    COO_to_CSR(B, csrB);
    COO_to_CSR(C, csrC);
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix* result = _raw_kernel_1_1(&csrB, &csrC);
    clock_gettime(CLOCK_MONOTONIC, &end);
    freeCSRMatrix(&csrB);
    freeCSRMatrix(&csrC);
    freeCSRMatrix(result);
    free(result);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return elapsed;
}

