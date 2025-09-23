#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdarg.h>

struct timespec start, end;

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

// Helper: swap two integers
static void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/* Generate a random CSR matrix
   rows, cols: matrix dimensions
   nnz_per_row: number of nonzeros per row
*/
CSRMatrix generate_random_CSR(int rows, int cols, int nnz_per_row) {
    if (nnz_per_row > cols) nnz_per_row = cols;  // clamp
    CSRMatrix M;
    M.rows = rows;
    M.cols = cols;
    M.nnz = nnz_per_row * rows;

    M.row_ptr = (int*)calloc(rows + 1, sizeof(int));
    M.col_ind = (int*)malloc(M.nnz * sizeof(int));
    M.val = (double*)malloc(M.nnz * sizeof(double));

    if (!M.row_ptr || !M.col_ind || !M.val) {
        perror("malloc");
        exit(1);
    }

    srand((unsigned)time(NULL));

    int nnz_counter = 0;
    for (int i = 0; i < rows; ++i) {
        M.row_ptr[i] = nnz_counter;

        // generate a random permutation of column indices and pick first nnz_per_row
        int *cols_perm = (int*)malloc(cols * sizeof(int));
        for (int c = 0; c < cols; ++c) {
            // printf("c: %d\n", c);
            cols_perm[c] = c;
        }

        for (int j = 0; j < nnz_per_row; ++j) {
            int r = j + rand() % (cols - j);
            swap(&cols_perm[j], &cols_perm[r]);
            M.col_ind[nnz_counter] = cols_perm[j];
            M.val[nnz_counter] = (double)rand() / RAND_MAX;  // random value [0,1)
            nnz_counter++;
        }

        free(cols_perm);
    }

    M.row_ptr[rows] = nnz_counter;
    return M;
}


// 1. Run the experiment (multiplyCSR) for increasing tensor dimension. Keep the sparsity per row as same for both matrix and same percentage of sparsity.
int experiment_1(int dimension_count, float sparsity)
{
    char filename[256];

    // create file name dynamically
    snprintf(filename, sizeof(filename), "exper1_dim%d_sparsity%.2f.csv",
             dimension_count, sparsity);
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("File opening failed");
        return 1;
    }
    fprintf(fp, "dimension,elapsed_time\n"); // CSV header


    for (int dimension = 4; dimension < dimension_count; dimension++)
    {
        int nnz_count_per_row = (int)round(dimension*sparsity);
        CSRMatrix B = generate_random_CSR(dimension, dimension, nnz_count_per_row);
        CSRMatrix C = generate_random_CSR(dimension, dimension, nnz_count_per_row);
        clock_gettime(CLOCK_MONOTONIC, &start);
        CSRMatrix A = multiplyCSR(&B, &C);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        // printf("Time: %.9f seconds\n", elapsed);
        fprintf(fp, "%d,%.9f\n", dimension, elapsed);
        freeCSRMatrix(&A);
        freeCSRMatrix(&B);
        freeCSRMatrix(&C);
    }

    fclose(fp);
    return 0;
}

int main(int argc, char *argv[]) {
    int dimension_count = atoi(argv[1]);
    float sparsity = atof(argv[2]);
    experiment_1(dimension_count, sparsity);
    return 0;
}

// int main() {
//     // Example: A = [ [1,0,2],
//     //                [0,3,0],
//     //                [4,0,5] ]
//     CSRMatrix A;
//     A.rows = 5; A.cols = 5; A.nnz = 6;
//     int Arp[] = {0,2,4,4,6,6};
//     int Aci[] = {2,4,2,3,1,2};
//     double Aval[] = {3,4,5,7,2,6};
//     // A.rows = 3; A.cols = 3; A.nnz = 5;
//     // int Arp[] = {0,2,3,5};
//     // int Aci[] = {0,2,1,0,2};
//     // double Aval[] = {1,2,3,4,5};
//     A.row_ptr = Arp; A.col_ind = Aci; A.val = Aval;

//     // B = [ [5,0,0],
//     //        [0,6,0],
//     //        [7,0,8] ]
//     CSRMatrix B;
//     B.rows = 5; B.cols = 5; B.nnz = 6;
//     int Brp[] = {0,0,2,4,6,6};
//     int Bci[] = {2,4,2,3,1,2};
//     double Bval[] = {3,4,5,7,2,6};
//     // B.rows = 3; B.cols = 3; B.nnz = 4;
//     // int Brp[] = {0,1,2,4};
//     // int Bci[] = {0,1,0,2};
//     // double Bval[] = {5,6,7,8};
//     B.row_ptr = Brp; B.col_ind = Bci; B.val = Bval;

//     CSRMatrix C = multiplyCSR(&A, &B);
//     printCSR(&C);
//     printCSRDense(&C);

//     free(C.row_ptr);
//     free(C.col_ind);
//     free(C.val);
//     return 0;
// }