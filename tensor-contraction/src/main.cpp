#include <formats.hpp>
#include "taco.h"
#include "taco_kernel.hpp"
#include "rand_gen.hpp"

using namespace taco;

int main(int argc, char *argv[])
{
    COOMatrix B = generate_synthetic_matrix(3, 3, 2);
    COOMatrix C = generate_synthetic_matrix(3, 3, 2);
    kernel_1_1(B, C);
    return 0;
}