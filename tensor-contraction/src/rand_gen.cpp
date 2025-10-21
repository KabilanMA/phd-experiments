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

COOMatrix generate_matrix_from_data(
    int rows,
    int cols,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices,
    const std::vector<double>& values
) {
    // Basic sanity checks
    if (row_indices.size() != col_indices.size() || row_indices.size() != values.size()) {
        throw std::invalid_argument("row_indices, col_indices, and values must have the same length");
    }

    COOMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_indices = row_indices;
    matrix.col_indices = col_indices;
    matrix.values = values;

    return matrix;
}

