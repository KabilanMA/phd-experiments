#include "taco_kernel.hpp"

struct timespec start, end;

// A(i,j) = B(i,j) * C(j,i)
// All are in CSR format initially
// Transpose C to CSC and transpose the tensor via "dense intermediate"
static double __taco_kernel_1_1(Tensor<double> &B, Tensor<double> &C, Tensor<double> &workspace);

double taco_kernel_1_1(const COOMatrix &B, const COOMatrix &C, Tensor<double> &workspace) 
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
    B_taco.pack();
    C_taco.pack();
    return __taco_kernel_1_1(B_taco, C_taco, workspace);
}

static double __taco_kernel_1_1(Tensor<double> &B, Tensor<double> &C, Tensor<double> &workspace) 
{
    Format csr({Dense, Sparse});
    Format csc({Sparse, Dense});
    Format dense_f({Dense, Dense});
    IndexVar i, j;

    Tensor<double> Ct_1({C.getDimension(1), C.getDimension(0)}, csc);
    Tensor<double> Ct_2({C.getDimension(1), C.getDimension(0)}, csc);
    Tensor<double> A({B.getDimension(0), B.getDimension(0)}, csr);
    
    // transpose_CSR_to_CSC_denseIntermediate(C, Ct_1);
    transpose_CSR_to_CSC_tacoInternals(C, Ct_2);
    // Ct.pack();
    
    A(i,j) = B(i,j) * C.transpose({1,0}, csc)(i,j);
    A.compile();   // generate code for the expression
    clock_gettime(CLOCK_MONOTONIC, &start);
    A.assemble();  // allocate indices, memory, etc.    
    A.compute();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    workspace = A;
    return elapsed;
}

