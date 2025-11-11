#include "taco_kernel.hpp"

struct timespec start, end;

// A(i,j) = B(i,j) * C(j,i)
// All are in CSR format initially
// Transpose C to CSC and transpose the tensor via "dense intermediate"
static double __taco_kernel_1_1(Tensor<double> &B, Tensor<double> &C, Tensor<double> &workspace);

// A(i,j) = B(i,k) * C(k,j)
// All are in CSR format initially
// Normal Matrix Multiplication
static double __taco_kernel_2_1(Tensor<double> &B, Tensor<double> &C, Tensor<double> &workspace);

// A(i, j) = B(i, k)  * C(k, j) * D(k, j)
static double __taco_kernel_3_1(Tensor<double> &B, Tensor<double> &C, Tensor<double> &D, Tensor<double> &workspace);

// A(i) = B(i, j) * C(j, i)
static double __taco_kernel_4_1(Tensor<double> &B, Tensor<double> &C, Tensor<double> &workspace);

// A(i) = B(i, j, k) * C(i, k, j)
// static double __taco_kernel_5_1(Tensor<double> &B, Tensor<double> &C, Tensor<double> &workspace);

// A(i) = B(i, j) * C(j, i)
double taco_kernel_4_1(const COOMatrix &B, const COOMatrix &C, const COOMatrix &D, Tensor<double> &workspace)
{
    Format csr({Dense, Sparse});
    Format csc({Sparse, Dense});
    Format dense_f({Dense, Dense});
    IndexVar i, j;

    Tensor<double> B_taco({B.rows, B.cols}, csr);
    generate_taco_tensor_from_coo(B, B_taco);
    Tensor<double> C_taco({C.rows, C.cols}, csr);
    generate_taco_tensor_from_coo(C, C_taco);
    Tensor<double> D_taco({D.rows, D.cols}, csr);
    generate_taco_tensor_from_coo(D, D_taco);

    B_taco.pack();
    C_taco.pack();
    D_taco.pack();

    double elapsed = __taco_kernel_4_1(B_taco, C_taco, D_taco, workspace);

    // std::cout << "=============== TACO Result ===============" << std::endl;
    // std::cout << workspace << std::endl;
    // std::cout << "===========================================" << std::endl;

    B_taco = Tensor<double>();
    C_taco = Tensor<double>();
    return elapsed;
}

// A(i) = B(i, j) * C(j, i)
static double __taco_kernel_4_1(Tensor<double> &B, Tensor<double> &C, Tensor<double> &D, Tensor<double> &workspace)
{
    struct timespec start, end;
    Format csr({Dense, Sparse});
    Format csc({Sparse, Dense});
    IndexVar i, j;

    Tensor<double> A({B.getDimension(0)}, {Dense});
    Tensor<double> Ct_2({C.getDimension(1), C.getDimension(0)}, csc);

    // Transpose first
    clock_gettime(CLOCK_MONOTONIC, &start);
    Ct_2 = C.transpose({1,0}, csc);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_1 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Define computation
    A(i) = sum(j, B(i,j) * Ct_2(i,j));

    // Compile and run
    A.compile();
    clock_gettime(CLOCK_MONOTONIC, &start);
    A.assemble();
    A.compute();
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_2 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    // workspace = A;

    return (elapsed_1 + elapsed_2);
}

// A(i, j) = B(i, k)  * C(k, j) * D(k, j)
double taco_kernel_3_1(const COOMatrix &B, const COOMatrix &C, const COOMatrix &D, Tensor<double> &workspace)
{
    Format csr({Dense, Sparse});
    Format csc({Sparse, Dense});
    Format dense_f({Dense, Dense});
    IndexVar i, j;

    Tensor<double> B_taco({B.rows, B.cols}, csr);
    generate_taco_tensor_from_coo(B, B_taco);
    Tensor<double> C_taco({C.rows, C.cols}, csr);
    generate_taco_tensor_from_coo(C, C_taco);
    Tensor<double> D_taco({D.rows, D.cols}, csr);
    generate_taco_tensor_from_coo(D, D_taco);

    B_taco.pack();
    C_taco.pack();
    D_taco.pack();

    double elapsed = __taco_kernel_3_1(B_taco, C_taco, D_taco, workspace);

    // std::cout << "=============== TACO Result ===============" << std::endl;
    // std::cout << workspace << std::endl;
    // std::cout << "===========================================" << std::endl;

    B_taco = Tensor<double>();
    C_taco = Tensor<double>();
    return elapsed;
}

// A(i, j) = B(i, k)  * C(k, j) * D(j, k)
static double __taco_kernel_3_1(Tensor<double> &B, Tensor<double> &C, Tensor<double> &D, Tensor<double> &workspace)
{
    Format csr({Dense, Sparse});
    Format csc({Sparse, Dense});
    Format dense_f({Dense, Dense});
    IndexVar i, j, k;

    Tensor<double> A({B.getDimension(0), C.getDimension(1)}, csr);

    Tensor<double> D_t({D.getDimension(1), D.getDimension(0)}, csr);
    Tensor<double> temp({C.getDimension(0), C.getDimension(1)}, csr);

    double elapsed = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &start);
    D_t = D.transpose({1,0}, csr);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    temp(k,j) = C(k,j) * D_t(k,j);
    // temp.compile();
    // clock_gettime(CLOCK_MONOTONIC, &start);
    // temp.assemble();
    // temp.compute();
    // clock_gettime(CLOCK_MONOTONIC, &end);
    // elapsed += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    A(i,j) = B(i,k) * temp(k,j);
    A.compile();
    clock_gettime(CLOCK_MONOTONIC, &start);
    A.assemble();
    A.compute();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    workspace = A;
    // std::cout << workspace << std::endl;
    D_t = Tensor<double>();
    // A = Tensor<double>();
    return (elapsed);
}

// Multiply A = B * C (MatMul)
// A(i, j) = B(i, k) * C(k, j)
double taco_kernel_2_1(const COOMatrix &B, const COOMatrix &C, Tensor<double> &workspace) 
{
    Format csr({Dense, Sparse});
    Format csc({Sparse, Dense});
    Format dense_f({Dense, Dense});
    IndexVar i, j;

    Tensor<double> B_taco({B.rows, B.cols}, csr);
    generate_taco_tensor_from_coo(B, B_taco);

    Tensor<double> C_taco({C.rows, C.cols}, csr);
    generate_taco_tensor_from_coo(C, C_taco);

    B_taco.pack();
    C_taco.pack();

    double elapsed = __taco_kernel_2_1(B_taco, C_taco, workspace);

    // std::cout << workspace << std::endl;
    
    B_taco = Tensor<double>();
    C_taco = Tensor<double>();
    return elapsed;
}

// Multiply A = B * C (MatMul)
// A(i, j) = B(i, k) * C(k, j)
static double __taco_kernel_2_1(Tensor<double> &B, Tensor<double> &C, Tensor<double> &workspace)
{
    Format csr({Dense, Sparse});
    Format csc({Sparse, Dense});
    Format dense_f({Dense, Dense});
    IndexVar i, j, k;

    Tensor<double> A({B.getDimension(0), C.getDimension(1)}, csr);

    A(i,j) = taco::sum(k, B(i,k) * C(k,j));
    A.compile();
    clock_gettime(CLOCK_MONOTONIC, &start);
    A.assemble();
    // std::cout << A.getSource() << std::endl;
    A.compute();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    workspace = A;
    return elapsed;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
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
    double elapsed = __taco_kernel_1_1(B_taco, C_taco, workspace);
    
    B_taco = Tensor<double>();
    C_taco = Tensor<double>();
    return elapsed;
}

// Multiply A = B * C (element wise row-col multiplication)
// A(i, j) = B(i, j) * C(j, i)
static double __taco_kernel_1_1(Tensor<double> &B, Tensor<double> &C, Tensor<double> &workspace) 
{
    Format csr({Dense, Sparse});
    Format csc({Sparse, Dense});
    Format dense_f({Dense, Dense});
    IndexVar i, j;

    Tensor<double> Ct_2({C.getDimension(1), C.getDimension(0)}, csc);
    Tensor<double> A({B.getDimension(0), B.getDimension(0)}, csr);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    Ct_2 = C.transpose({1,0}, csc);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed1 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    A(i,j) = B(i,j) * Ct_2(i,j);
    A.compile();   // generate code for the expression
    clock_gettime(CLOCK_MONOTONIC, &start);
    A.assemble();  
    A.compute();
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed2 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    workspace = A;
    std::cout << workspace << std::endl;
    Ct_2 = Tensor<double>();
    A = Tensor<double>();
    return (elapsed1+elapsed2);
}

