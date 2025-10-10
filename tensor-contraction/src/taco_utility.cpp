#include "taco_utility.hpp"

using namespace taco;

// Transpose a CSR matrix A into a CSC matrix At using a dense intermediate representation
// A should be in CSR format and At should be in CSC format
// Both A and At should be pre-allocated with the correct dimensions and formats
void transpose_CSR_to_CSC_denseIntermediate(Tensor<double> &A, Tensor<double> &At) {
    IndexVar i, j;
    Tensor<double> At_dense({A.getDimension(0), A.getDimension(1)}, {Dense, Dense});
    At_dense(j, i) = A(i, j);
    At_dense.compile();
    At_dense.assemble();
    At_dense.compute();
    
    // Tensor<double> *At = new Tensor<double>({A.getDimension(0), A.getDimension(1)}, {Sparse, Dense});
    At(i,j) = At_dense(i,j);
    At.compile();
    At.assemble();
    At.compute();
}

// A is in CSR and At is in CSC format
void transpose_CSR_to_CSC_tacoInternals(Tensor<double> &A, Tensor<double> &At) {
    // Tensor<double> A_coo({A.getDimension(0), A.getDimension(1)}, {Dense, Dense});
    // IndexVar i, j;

    // A_coo(i,j) = A(i,j);
    // // A_coo.evaluate();
    // A_coo.compile();
    // A_coo.assemble();
    // A_coo.compute();

    At = A.transpose({1,0}, {Sparse, Dense});
}

void generate_taco_tensor_from_coo(const COOMatrix &coo, Tensor<double> &tensor)
{
    for (int idx = 0; idx < coo.values.size(); idx++) {
        tensor.insert({coo.row_indices[idx], coo.col_indices[idx]}, coo.values[idx]);
    }
    // taco::COO(2, true, true, false, {});
    return;
}