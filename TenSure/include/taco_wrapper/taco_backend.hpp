// // include/taco_wrapper/taco_backend.hpp
// #pragma once
// #include "backend_interface.hpp"
// #include "tensure/formats.hpp"

// struct TacoBackend : public FuzzBackend {
//     bool generate_kernel(const std::vector<tsTensor>& tensors,
//                          const std::vector<std::string>& computations,
//                          const std::vector<std::string>& dataFiles,
//                          const std::string& outFile) override;

//     bool execute_kernel(const std::string& kernelPath,
//                         const std::string& outputDir) override;

//     bool compare_results(const std::string& refDir,
//                          const std::string& testDir) override;
// };