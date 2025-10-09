#include "formats.hpp"

void freeCSRMatrix(CSRMatrix *M) 
{
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

void printCSR(const CSRMatrix *M) 
{
    printf("CSR Matrix: %d x %d, nnz = %d\n", M->rows, M->cols, M->nnz);
    printf("row_ptr: ");
    for (int i = 0; i <= M->rows; i++) printf("%d ", M->row_ptr[i]);
    printf("\ncol_ind: ");
    for (int i = 0; i < M->nnz; i++) printf("%d ", M->col_ind[i]);
    printf("\nval:     ");
    for (int i = 0; i < M->nnz; i++) printf("%.2f ", M->val[i]);
    printf("\n");
}

void printCOO(const COOMatrix &M)
{
    printf("COO Matrix: %d x %d, nnz = %d\n", M.rows, M.cols, (int)M.values.size());
    printf("row_indices: ");
    for (size_t i = 0; i < M.row_indices.size(); i++) {
        printf("%d ", M.row_indices[i]);
    }
    printf("\ncol_indices: ");
    for (size_t i = 0; i < M.col_indices.size(); i++) {
        printf("%d ", M.col_indices[i]);
    }
    printf("\nvalues: ");
    for (size_t i = 0; i < M.values.size(); i++) {
        printf("%.2f ", M.values[i]);
    }
    printf("\n");
}

void printCOODense(const COOMatrix &M)
{
    for (int i =0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {
            double val = 0.0;
            for (size_t idx = 0; idx < M.row_indices.size(); idx++)
            {
                if (M.row_indices[idx] == i && M.col_indices[idx] == j)
                {
                    val = M.values[idx];
                    break;
                }
            }
            printf("%6.2f ", val);
        }
        printf("\n");
    }
}

void printCSRDense(const CSRMatrix *M) 
{
    if (!M) return;

    for (int i = 0; i < M->rows; i++) 
    {
        int row_start = M->row_ptr[i];
        int row_end   = M->row_ptr[i + 1];

        // Pointer into current row's CSR entries
        int csr_index = row_start;

        for (int j = 0; j < M->cols; j++) 
        {
            double val = 0.0;
            if (csr_index < row_end && M->col_ind[csr_index] == j) 
            {
                val = M->val[csr_index];
                csr_index++;
            }
            printf("%6.2f ", val);
        }
        printf("\n");
    }
}

static void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

CSRMatrix generate_random_CSR(int rows, int cols, int nnz_per_row) 
{
    if (nnz_per_row > cols) nnz_per_row = cols;  // clamp
    CSRMatrix M;
    M.rows = rows;
    M.cols = cols;
    M.nnz = nnz_per_row * rows;

    M.row_ptr = (int*)calloc(rows + 1, sizeof(int));
    M.col_ind = (int*)malloc(M.nnz * sizeof(int));
    M.val = (double*)malloc(M.nnz * sizeof(double));

    if (!M.row_ptr || !M.col_ind || !M.val) 
    {
        perror("malloc");
        exit(1);
    }

    srand((unsigned)time(NULL));

    int nnz_counter = 0;
    for (int i = 0; i < rows; ++i) 
    {
        M.row_ptr[i] = nnz_counter;

        // generate a random permutation of column indices and pick first nnz_per_row
        int *cols_perm = (int*)malloc(cols * sizeof(int));
        for (int c = 0; c < cols; ++c) 
        {
            // printf("c: %d\n", c);
            cols_perm[c] = c;
        }

        for (int j = 0; j < nnz_per_row; ++j) 
        {
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

void COO_to_CSR(const COOMatrix &coo, CSRMatrix &csr)
{
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.values.size();

    csr.row_ptr = (int *)calloc((csr.rows + 1), sizeof(int));
    csr.col_ind = (int *)malloc(csr.nnz * sizeof(int));
    csr.val = (double *)malloc(csr.nnz * sizeof(double));

    // Count the number of non-zeros in each row
    for (int i = 0; i < csr.nnz; i++) {
        csr.row_ptr[coo.row_indices[i] + 1]++;
    }

    // Compute the prefix sum to get row_ptr
    for (int i = 0; i < csr.rows; i++) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    // Fill col_ind and val arrays
    std::vector<int> counter(csr.rows, 0);
    for (int i = 0; i < csr.nnz; i++) {
        int row = coo.row_indices[i];
        int dest = csr.row_ptr[row] + counter[row];
        csr.col_ind[dest] = coo.col_indices[i];
        csr.val[dest] = coo.values[i];
        counter[row]++;
    }
}