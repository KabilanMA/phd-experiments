// #include "taco_wrapper/generator.hpp"
// #include "tensure/logger.hpp"

// bool generate_kernel(const std::vector<tsTensor> &tensors,
//                      const std::vector<std::string> &computations,
//                      const std::vector<std::string> &data_files,
//                      const std::string &output_json_path,
//                      FuzzBackend *backend)
// {
//     Logger log;

//     if (!backend) {
//         log.error("generate_kernel: No backend loaded!");
//         return false;
//     }

//     try {
//         bool ok = backend->generate_kernel(tensors, computations, data_files, output_json_path);
//         if (!ok) {
//             log.error("Backend failed to generate kernel.");
//             return false;
//         }

//         log.info("Kernel generated successfully at " + output_json_path);
//         return true;
//     }
//     catch (const std::exception &e) {
//         log.error(std::string("Exception in generate_kernel: ") + e.what());
//         return false;
//     }
// }

// bool generate_kernel_taco(vector<tsTensor>& tensors,
//                      vector<string> computations,
//                      vector<string> dataFileNames,
//                      const string& file_name)
// {
//     if (tensors.size() != dataFileNames.size()) return false;

//     tsKernel kernel;
//     for (size_t i = 0; i < tensors.size(); i++) {
//         auto &tensor = tensors[i];
//         kernel.tensors.push_back(tensor);
//         kernel.dataFileNames.insert({string(1, tensor.name), dataFileNames[i]});
//     }

//     for (auto& computation : computations) {
//         tsComputation comp;
//         comp.expressions = computation;
//         kernel.computations.push_back(comp);
//     }

//     // ---- Atomic write ----
//     std::string tmp_name = file_name + ".tmp";

//     try {
//         kernel.saveJson(tmp_name);      // write to temporary file first
//         std::filesystem::rename(tmp_name, file_name); // atomic replacement
//     } catch (const std::exception& e) {
//         std::cerr << "generate_kernel failed: " << e.what() << std::endl;
//         // Cleanup temp if partially written
//         std::error_code ec;
//         std::filesystem::remove(tmp_name, ec);
//         return false;
//     }

//     return true;
// }
