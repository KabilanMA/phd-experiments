#include "formats.hpp"

double raw_kernel_1_1(const COOMatrix &B, const COOMatrix &C);
double raw_kernel_2_1(const COOMatrix &B, const COOMatrix &C);
CSRMatrix* raw_kernel_3_1(const COOMatrix &B, const COOMatrix &C, const COOMatrix &D);