#pragma once
#include <formats.hpp>

COOMatrix generate_synthetic_matrix(int rows, int cols, int nnz_per_row);
COOMatrix generate_matrix_from_data(
    int rows,
    int cols,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices,
    const std::vector<double>& values
);