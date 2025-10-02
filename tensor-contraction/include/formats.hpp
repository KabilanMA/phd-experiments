#ifndef FORMATS_HPP
#define FORMATS_HPP

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdarg.h>

typedef struct COOMatrix {
    int rows;
    int cols;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;
} COOMatrix;

typedef struct CSCMatrix {
    int rows;
    int cols;
    int nnz;       // number of non-zeros
    int *col_ptr;  // size = cols + 1
    int *row_ind;  // size = nnz
    double *val;   // size = nnz
} CSCMatrix;

typedef struct {
    int rows;
    int cols;
    int nnz;       // number of non-zeros
    int *row_ptr;  // size = rows + 1
    int *col_ind;  // size = nnz
    double *val;   // size = nnz
} CSRMatrix;

#endif // FORMATS_HPP
