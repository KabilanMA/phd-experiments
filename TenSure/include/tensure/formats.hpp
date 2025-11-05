#pragma once
#include <string>
#include <vector>
#include <sstream> 
#include <fstream> 
#include <nlohmann/json.hpp>

using namespace std;

enum class MutationOperator {
    SPARSITY,
    DIMENSION,
    DATATYPE,
    FORMAT,
    VALUE
};

enum tsFormat {
    tsSparse=0,
    tsDense=1
};

enum tsDataType {
    INT = 0,
    DOUBLE = 1,
    FLOAT = 2
};

typedef struct tacoTensor
{
    char name;
    std::string str_repr;
    std::vector<char> idxs;
    std::vector<std::string> storageFormat;
} tacoTensor;


typedef struct tsTensor {
    string name;                 // Tensor name, e.g., "A"
    vector<string> axes;
    vector<int> shape;           // Dimensions, e.g., [3, 3]
    vector<tsFormat> format;
    tsDataType dtype;                // "float", "double", etc.
    string data_path;            // Path to file holding tensor data
    string str_repr;

} tsTensor;

typedef struct tsComputation {
    vector<string> expressions;
} tsComputation;

typedef struct tsKernel {
    string name;
    vector<tsTensor> tensors;
    tsComputation computations;
    string backend_name;
    string json_path;

    // --- Internal utility functions ---
    // void loadJson(const std::string &path)
    // {
    //     std::ifstream in(path);
    //     if (!in.is_open()) throw std::runtime_error("Failed to open kernel JSON: " + path);

    //     nlohmann::json j;
    //     in >> j;

    //     name = j.at("name").get<std::string>();
    //     backend_name = j.at("backend_name").get<std::string>();
    //     json_path = j.at("json_path").get<std::string>();
    //     computations = j.at("computations").get<std::vector<std::string>>();
    //     tensors.clear();

    //     for (const auto &jt : j.at("tensors")) {
    //         tensors.push_back(tsTensor::from_json(jt));
    //     }  
    // }

    // void saveJson(const std::string &path) const
    // {
    //     nlohmann::json j;
    //     j["name"] = name;
    //     j["backend_name"] = backend_name;
    //     j["json_path"] = json_path;
    //     j["computations"] = computations;
    //     j["tensors"] = nlohmann::json::array();

    //     for (const auto &t : tensors) {
    //         nlohmann::json jt;
    //         jt["name"] = t.name;
    //         jt["shape"] = t.shape;
    //         jt["format"] = t.format;
    //         jt["dtype"] = t.dtype;
    //         jt["data_path"] = t.data_path;
    //         jt["is_input"] = t.is_input;
    //         j["tensors"].push_back(jt);
    //     }

    //     std::ofstream out(path);
    //     if (!out.is_open()) throw std::runtime_error("Failed to open kernel JSON: " + path);
    //     out << j.dump(4);
    // }
    // std::string summary() const
    // {
    //     std::ostringstream oss;
    //     oss << "Kernel: " << name << " [backend: " << backend_name << "]\n";
    //     oss << "Tensors:\n";
    //     for (const auto &t : tensors) {
    //         oss << "  - " << t.str() << " (data: " << t.data_path << ")\n";
    //     }
    //     oss << "Computations:\n";
    //     for (const auto &c : computations) {
    //         oss << "  - " << c << "\n";
    //     }
    //     return oss.str();
    // }
} tsKernel;
