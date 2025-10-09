#pragma once
#include "formats.hpp"
#include "taco.h"

using namespace taco;

void transpose_CSR_to_CSC_denseIntermediate(Tensor<double> &A, Tensor<double> &At);
void transpose_CSR_to_CSC_tacoInternals(Tensor<double> &A, Tensor<double> &At);

void generate_taco_tensor_from_coo(const COOMatrix &coo, Tensor<double> &tensor);