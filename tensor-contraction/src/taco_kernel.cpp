#include "taco_kernel.hpp"

// void generate_taco_tensor_from_coo(COOMatrix &coo, Tensor<double> &tensor)
// {
//     for (int idx = 0; idx < coo.values.size(); idx++) {
//         tensor.insert({coo.row_indices[idx], coo.col_indices[idx]}, coo.values[idx]);
//     }
//     return;
// }

// A(i,j) = B(i,j) * C(j,i)
// All are in CSR format initially
// Transpose C to CSC and transpose the tensor via "dense intermediate"
void kernel_1_1(COOMatrix &B, COOMatrix &C) 
{
    Format csr({Dense, Sparse});
    Format csc({Sparse, Dense});
    Format dense_f({Dense, Dense});
    IndexVar i, j;

    Tensor<double> B_taco({B.rows, B.cols}, csr);
    generate_taco_tensor_from_coo(B, B_taco);
    // B_taco.pack();

    Tensor<double> C_taco({C.rows, C.cols}, csr);
    generate_taco_tensor_from_coo(C, C_taco);
    // C_taco.pack();
    Tensor<double> Ct({C.cols, C.rows}, csc);
    transpose_CSR_to_CSC_denseIntermediate(C_taco, Ct);
    // Ct.pack();
    
    Tensor<double> A({B.rows, C.cols}, csr);
    A(i,j) = B_taco(i,j) * Ct(i,j);

    std::cout << A << std::endl;
}
