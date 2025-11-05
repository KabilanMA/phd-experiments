// #include "backend_interface.h"
// #include <iostream>

// // You already have these implemented elsewhere
// bool generate_kernel_taco(const std::vector<std::string>& tensors,
//                           const std::vector<std::string>& computations,
//                           const std::vector<std::string>& dataFiles,
//                           const std::string& outFile);

// bool execute_kernel_taco(const std::string& kernelPath,
//                          const std::string& outputDir);

// bool compare_results_taco(const std::string& refDir,
//                           const std::string& testDir);

// // Actual plugin class implementing the abstract interface
// struct TacoBackend : public FuzzBackend {
//     bool generate_kernel(const std::vector<std::string>& tensors,
//                          const std::vector<std::string>& computations,
//                          const std::vector<std::string>& dataFiles,
//                          const std::string& outFile) override {
//         return generate_kernel_taco(tensors, computations, dataFiles, outFile);
//     }

//     bool execute_kernel(const std::string& kernelPath,
//                         const std::string& outputDir) override {
//         return execute_kernel_taco(kernelPath, outputDir);
//     }

//     bool compare_results(const std::string& refDir,
//                          const std::string& testDir) override {
//         return compare_results_taco(refDir, testDir);
//     }
// };

// // Plugin entry points required by the loader
// extern "C" FuzzBackend* create_backend() {
//     return new TacoBackend();
// }

// extern "C" void destroy_backend(FuzzBackend* backend) {
//     delete backend;
// }
