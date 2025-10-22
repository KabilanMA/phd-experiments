#pragma once
#include <formats.hpp>
#include "taco.h"
#include "taco_utility.hpp"

using namespace taco;

double taco_kernel_1_1(const COOMatrix &B, const COOMatrix &C, Tensor<double> &workspace);
double taco_kernel_2_1(const COOMatrix &B, const COOMatrix &C, Tensor<double> &workspace);
double taco_kernel_3_1(const COOMatrix &B, const COOMatrix &C, const COOMatrix &D, Tensor<double> &workspace);
double taco_kernel_4_1(const COOMatrix &B, const COOMatrix &C, Tensor<double> &workspace);

