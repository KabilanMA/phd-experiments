#ifndef FORMATS_HPP
#define FORMATS_HPP

#include "uthash.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <stdarg.h>
#include <string.h>
#include <iomanip>
#include <algorithm>
// #include <linux/time.h>

typedef struct COOMatrix {
    int rows;
    int cols;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;

    void print() const {
        std::cout << "COO Matrix (" << rows << "x" << cols << ")\n";
        std::cout << "Row  Col  Value\n";
        for (size_t i = 0; i < values.size(); ++i) {
            std::cout << std::setw(3) << row_indices[i] << "  "
                 << std::setw(3) << col_indices[i] << "  "
                 << values[i] << "\n";
        }
        std::cout << std::endl;
    }

    void print_dense() const {
        std::vector<std::vector<double>> dense(rows, std::vector<double>(cols, 0.0));
        for (size_t i = 0; i < values.size(); ++i) {
            dense[row_indices[i]][col_indices[i]] = values[i];
        }

        std::cout << "COO Dense view (" << rows << "x" << cols << "):\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                std::cout << std::setw(6) << dense[i][j] << " ";
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
    
} COOMatrix;

typedef struct CSCMatrix {
    int rows;
    int cols;
    int nnz;       // number of non-zeros
    int *col_ptr;  // size = cols + 1
    int *row_ind;  // size = nnz
    double *val;   // size = nnz

    CSCMatrix() : rows(0), cols(0), nnz(0), col_ptr(nullptr), row_ind(nullptr), val(nullptr) {}

    void print() const {
        std::cout << "CSC Matrix (" << rows << "x" << cols << "), nnz=" << nnz << "\n";
        std::cout << "col_ptr: ";
        for (int i = 0; i < cols + 1; ++i) std::cout << col_ptr[i] << " ";
        std::cout << "\nrow_ind: ";
        for (int i = 0; i < nnz; ++i) std::cout << row_ind[i] << " ";
        std::cout << "\nval:     ";
        for (int i = 0; i < nnz; ++i) std::cout << val[i] << " ";
        std::cout << "\n\n";
    }

    void print_dense() const {
        std::vector<std::vector<double>> dense(rows, std::vector<double>(cols, 0.0));
        for (int j = 0; j < cols; ++j) {
            for (int idx = col_ptr[j]; idx < col_ptr[j+1]; ++idx) {
                dense[row_ind[idx]][j] = val[idx];
            }
        }

        std::cout << "CSC Dense view (" << rows << "x" << cols << "):\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                std::cout << std::setw(6) << dense[i][j] << " ";
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    void free() {
        delete[] col_ptr;
        delete[] row_ind;
        delete[] val;
        col_ptr = nullptr;
        row_ind = nullptr;
        val = nullptr;
    }

} CSCMatrix;

typedef struct CSRMatrix {
    int rows;
    int cols;
    int nnz;       // number of non-zeros
    int *row_ptr;  // size = rows + 1
    int *col_ind;  // size = nnz
    double *val;   // size = nnz

    CSRMatrix() : rows(0), cols(0), nnz(0), row_ptr(nullptr), col_ind(nullptr), val(nullptr) {}

    void print() const {
        std::cout << "CSR Matrix (" << rows << "x" << cols << "), nnz=" << nnz << "\n";
        std::cout << "row_ptr: ";
        for (int i = 0; i < rows + 1; ++i) std::cout << row_ptr[i] << " ";
        std::cout << "\ncol_ind: ";
        for (int i = 0; i < nnz; ++i) std::cout << col_ind[i] << " ";
        std::cout << "\nval:     ";
        for (int i = 0; i < nnz; ++i) std::cout << val[i] << " ";
        std::cout << "\n\n";
    }

    void print_dense() const {
        std::vector<std::vector<double>> dense(rows, std::vector<double>(cols, 0.0));
        for (int i = 0; i < rows; ++i) {
            for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
                dense[i][col_ind[j]] = val[j];
            }
        }

        std::cout << "CSR Dense view (" << rows << "x" << cols << "):\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                std::cout << std::setw(6) << dense[i][j] << " ";
            std::cout << "\n";
        }
        std::cout << std::endl;
    }

    void free() {
        delete[] row_ptr;
        delete[] col_ind;
        delete[] val;
        row_ptr = nullptr;
        col_ind = nullptr;
        val = nullptr;
    }

} CSRMatrix;

void COO_to_CSR(const COOMatrix &coo, CSRMatrix &csr);
void COO_to_CSC(const COOMatrix &coo, CSCMatrix &csc);
void freeCSRMatrix(CSRMatrix *M);
void freeCOOMatrix(COOMatrix *M);
void freeCSCMatrix(CSCMatrix *M);
CSRMatrix generate_random_CSR(int rows, int cols, int nnz_per_row);
void printCSRDense(const CSRMatrix *M);
void printCSR(const CSRMatrix *M);
void printCOO(const COOMatrix &M);
void printCOODense(const COOMatrix &M);

#endif // FORMATS_HPP
