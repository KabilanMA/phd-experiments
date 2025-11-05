#pragma once
#include <string>
#include <vector>
#include "tensure/formats.hpp"

struct FuzzBackend {
    virtual ~FuzzBackend() = default;

    // Generate kernel source code for a given einsum specification.
    virtual bool generate_kernel(const std::vector<tsTensor> &tensors,
                                 const std::vector<std::string> &computations,
                                 const std::vector<std::string> &data_files,
                                 const std::string &output_json_path) = 0;

    // Optional extensions (you likely have them)
    virtual bool compile_kernel(const std::string &source_path) { return true; }
    virtual bool run_kernel(const std::string &binary_path) { return true; }
};