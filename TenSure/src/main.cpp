#include "taco_generator.hpp"
#include "taco.h"

using namespace taco;
int main(int argc, char * argv[])
{
    auto [tensors, einsum] = random_valid_einsum(2,3);
    generate_store_tensor_data(tensors, "12", "");
    // generate_taco_kernel(tensors, einsum);
    // std::cout << std::string(3, 'D') << std::endl;
    // for (auto &tensor : tensors)
    // {
    //     std::cout << tensor << std::endl;
    // }
    // std::cout << einsum << std::endl;

    
    // IndexVar i, j, k, l;
    // std::vector<ModeFormat> fmt;
    // fmt.push_back(Dense);
    // fmt.push_back(Dense);
    // fmt.push_back(Dense);
    // // Define tensors
    // Tensor<double> A("A", {2, 2, 2}, {Dense, Dense, Dense});
    // Tensor<double> B("B", {2, 2}, {Dense, Dense});
    // Tensor<double> C("C", {2, 2, 2}, Format{Dense, Dense, Dense});

    // // Fill B and C with some sample values
    // for (int i_ = 0; i_ < 2; i_++) {
    //     for (int j_ = 0; j_ < 2; j_++) {
    //         B.insert({i_, j_}, i_ + j_ + 1.0);
    //         for (int k_ = 0; k_ < 2; k_++) {
    //             C.insert({i_, j_, k_}, i_ + j_ + k_ + 0.5);
    //         }
    //     }
    // }

    // B.pack();
    // C.pack();

    // // Compute A(i,j,k) = sum_l B(i,l) * C(l,j,k)
    // A(i, j, k) = B(i, l) * C(l, j, k);
    // A.compile();
    // A.assemble();
    // A.compute();

    // std::cout << A << std::endl;

    // write("b.tns", B);

    return 0;
}