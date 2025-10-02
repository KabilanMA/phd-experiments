#include "rand_gen.hpp"

COOMatrix generate_synthetic_matrix(int rows, int cols, int nnz_per_row) {
    COOMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    for (int i = 0; i < rows; i++) {
        int count = 0;
        for (int j =0; j < cols; j++) {
            if (count < nnz_per_row && (rand() % cols) < nnz_per_row)
            {
                matrix.row_indices.push_back(i);
                matrix.col_indices.push_back(j);
                matrix.values.push_back((double)rand() / RAND_MAX);
                count++;
            }
        }
    }

    return matrix;
}

