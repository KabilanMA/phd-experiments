#include "taco_generator.hpp"

/**
 * Utility: Overload the output stream operator for tacoTensor.
 * @param os Output stream
 * @param tensor tacoTensor to print
 * @return Output stream
 */
std::ostream& operator<<(std::ostream& os, const tacoTensor& tensor) 
{
    os << tensor.str_repr ;
    return os;
}

/**
 * Find all unique indices in the given taco tensors.
 * @param taco_tensors Vecotor of tacoTensor
 * @return Set of unique indices
 */
std::set<char> find_idxs(std::vector<tacoTensor> taco_tensors)
{
    std::set<char> idxs;
    for (auto &taco_tensor : taco_tensors)
    {
        for (auto &idx : taco_tensor.idxs)
        {
            idxs.insert(idx);
        }
    }
    return idxs;
}

/**
 * Map each index to a random value between 3 and 20.
 * This is used to define the dimensions of the tensors.
 * @param idxs Set of indices
 * @return Map of index to random value
 */
std::map<char, int> map_id_to_val(std::set<char> idxs)
{
    std::map<char, int> id_val_map;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(3, 20);
    for (auto &idx: idxs)
    {
        id_val_map[idx] = dist(gen);
    }

    return id_val_map;
}

/**
 * Utility: Randomly return whether a tensor dimension is sparse or dense.
 * @param gen Random number generator
 * @return TensorFormat (tSparse or tDense)
 */
TensorFormat randomFormat(std::mt19937& gen) {
    std::uniform_int_distribution<int> dist(0, 1);
    return dist(gen) ? tSparse : tDense;
}

// Utility: join indices as string (e.g., i,j,k)
std::string join(const std::vector<char>& idxs) {
    std::string s;
    for (size_t i = 0; i < idxs.size(); ++i) {
        s += idxs[i];
        if (i + 1 < idxs.size()) s += ",";
    }
    return s;
}

std::vector<std::string> split(const std::string& s, char delimiter = ',') {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        tokens.push_back(item);
    }
    return tokens;
}

/**
 * Utility: Join strings with commas as the separator.
 * @param idxs Vector of strings
 * @return Comma-separated string
 */
std::string join(const std::vector<std::string>& idxs) {
    std::string s;
    for (size_t i = 0; i < idxs.size(); ++i) {
        s += idxs[i];
        if (i + 1 < idxs.size()) s += ",";
    }
    return s;
}

taco::Format makeFormat(const std::vector<std::string>& formatStrings) {
    std::vector<taco::ModeFormatPack> modeFormats;
    modeFormats.reserve(formatStrings.size());

    for (const auto& s : formatStrings) {
        if (s == "Dense")
            modeFormats.push_back(taco::Dense);
        else if (s == "Sparse")
            modeFormats.push_back(taco::Sparse);
        else
            throw std::invalid_argument("Unknown format type: " + s);
    }

    return taco::Format(modeFormats);
}

// Save
void saveTensor(const tacoTensor& t, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Failed to open file for writing");

    out << t.name << "\n";
    out << t.str_repr << "\n";

    // Save idxs
    out << t.idxs.size() << " ";
    for (char c : t.idxs) out << c << " ";
    out << "\n";

    // Save storage format
    out << t.storageFormat.size() << " ";
    for (auto& s : t.storageFormat) out << s << " ";
    out << "\n";
}

// Load
tacoTensor loadTensor(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Failed to open file for reading");

    tacoTensor t;
    in >> t.name;
    in.ignore(); // eat newline
    std::getline(in, t.str_repr);

    size_t idxCount;
    in >> idxCount;
    t.idxs.resize(idxCount);
    for (size_t i = 0; i < idxCount; i++) in >> t.idxs[i];

    size_t fmtCount;
    in >> fmtCount;
    t.storageFormat.resize(fmtCount);
    for (size_t i = 0; i < fmtCount; i++) in >> t.storageFormat[i];

    return t;
}

// Recursive helper to simulate N nested loops
void fillTensorRecursive(taco::Tensor<double>& T, 
                         const std::vector<int>& dims,
                         std::vector<int>& current, std::mt19937 gen, std::bernoulli_distribution insert_dist, int depth) {
    if (depth == dims.size()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 5.0);
        double randomValue = dist(gen);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << randomValue;
        std::string strValue = oss.str();

        // double val = std::round(dist(gen) * 100.0) / 100.0;
        if (insert_dist(gen))
            T.insert(current, stod(strValue));
        return;
    }
    for (int i = 0; i < dims[depth]; i++) {
        current[depth] = i;
        fillTensorRecursive(T, dims, current, gen, insert_dist, depth + 1);
    }
}

static void generate_random_tensor_data(taco::Tensor<double>& tensor, const std::vector<int>& tensor_dimension)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution insert_dist(0.4);

    std::vector<int> current(tensor_dimension.size());
    fillTensorRecursive(tensor, tensor_dimension, current, gen, insert_dist, 0);
    
}

bool generate_store_tensor_data(const std::vector<tacoTensor>& taco_tensors, std::string file_name_suffix, std::string location)
{
    std::set<char> all_idxs = find_idxs(taco_tensors);
    std::map<char, int> id_val_map = map_id_to_val(all_idxs);
    for (auto &tensor : taco_tensors)
    {
        std::vector<int> tensor_dimension;
        for (auto &idx : tensor.idxs)
        {
            tensor_dimension.push_back(id_val_map[idx]);    
        }

        // std::cout << makeFormat(tensor.storageFormat) << std::endl;

        taco::Tensor<double> A(std::string(1, tensor.name), tensor_dimension, makeFormat(tensor.storageFormat));

        generate_random_tensor_data(A, tensor_dimension);

        std::string tensor_data_file_name = std::string(1, tensor.name) + "_" + file_name_suffix + ".tns";

        taco::write(tensor_data_file_name, A);
        saveTensor(tensor, (std::string(1, tensor.name) + ".txt"));
        tacoTensor B = loadTensor((std::string(1, tensor.name) + ".txt"));
        saveTensor(B, (std::string(1, B.name) + "2.txt"));
        // std::cout << A << std::endl;
        
        // var_init += (std::string("") + "Tensor<double> " + tensor.name + "(\"" + tensor.name + "\", {" + join(format_str) + "}, Format({" + tensor.storageFormat + "});\n\t");
    }

    std::cout << "AFSD" << std::endl;

    return true;
}

static std::string generate_headers()
{
    std::string headers = "";
    headers += "#include \"taco.h\"\n\n";
    headers += "using namespace taco;\n\n";
    return headers;
}

static std::string generate_main_start(std::string& full_program)
{
    std::string main_start = "";
    main_start += "int main(int argc, char * argv[])\n{\n";
    main_start += full_program + "\n}\n";
    return main_start;
}

static std::string generate_taco_var_init(const std::vector<tacoTensor>& taco_tensors, std::set<char>& all_idxs, std::map<char , int>& id_val_map)
{
    std::string var_init = "";
    var_init += "\tIndexVar ";

    // Declare all iterator index variables in TACO
    for (auto it = all_idxs.begin(); it != all_idxs.end(); ++it)
    {
        var_init += *it;
        if (std::next(it) != all_idxs.end())
            var_init += ", ";
    }
    var_init += ";\n\n\t";

    for (auto &tensor : taco_tensors)
    {
        std::vector<std::string> format_str;
        for (auto &idx : tensor.idxs)
        {
            format_str.push_back(std::to_string(id_val_map[idx]));         
        }
        // var_init += (std::string("") + "Tensor<double> " + tensor.name + "(\"" + tensor.name + "\", {" + join(format_str) + "}, Format({" + tensor.storageFormat + "});\n\t");
    }
    var_init += "\n";
    return var_init;
}

static std::string generate_taco_tensor_inserts(const std::vector<tacoTensor>& taco_tensors, std::map<char, int>& id_val_map)
{
    std::string tensor_inserts = "";

    for(auto &tensor : taco_tensors)
    {
        if (tensor.name != 'A') // assuming A is always the output tensor
        {
            std::string for_loop = "";
            std::vector<std::string> format_str;
            for (size_t i =0; i < tensor.idxs.size(); i++)
            {
                char idx = tensor.idxs[i];
                std::string loop_index = std::string(2, idx);
                for_loop += "for (int " + loop_index + " = 0; " + loop_index + " < " + std::to_string(id_val_map[idx]) + "; " + loop_index + "++)\n" + std::string(8*(i+2), ' ');
                // std::cout << for_loop << std::endl;
                format_str.push_back(loop_index);
                if (i == tensor.idxs.size() - 1)
                {
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<> dist(0.0, 5.0);
                    double randomValue = dist(gen);

                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << randomValue;
                    std::string strValue = oss.str();

                    for_loop += std::string(1, tensor.name) + ".insert({" + join(format_str) + "}, " + strValue + ");";
                }
            }
            tensor_inserts += for_loop + "\n\n\t";
        }
    }

    return tensor_inserts;
}

// std::vector<int> generate
// std::string& generate_taco_kernel(std::vector<tacoTensor> taco_tensors, std::string einsum_rhs)
// {
//     std::set<char> all_idxs = find_idxs(taco_tensors);
//     std::map<char, int> id_val_map = map_id_to_val(all_idxs);
//     for(auto &tensor : taco_tensors)
//     {
//         std::vector<int> dimension;
//         for (auto &idx : tensor.idxs)
//         {
//             dimension.push_back(id_val_map[idx]);         
//         }


//         // taco::Tensor<double> A(std::string(1, tensor.name), {2,2,2}, taco::Format{taco::Dense, taco::Sparse, taco::Sparse})
//     }


//     static int kernel_id = 0;
//     static std::string kernel_code = "";
//     kernel_code += "#include <iostream>\n#include <random>\n#include \"taco.h\"\n\n";
    
//     kernel_code += "using namespace taco;\n\n";
//     kernel_code += "int main(int argc, char * argv[])\n{\n\t";
//     kernel_code += "std::random_device rd;\n\tstd::mt19937 gen(rd());\n\tstd::bernoulli_distribution insert_dist(0.5);\n\n\t";
//     kernel_code += "IndexVar ";
    
    
//     // Declare all iterator index variables in TACO
//     for (auto it = all_idxs.begin(); it != all_idxs.end(); ++it)
//     {
//         kernel_code += *it;
//         if (std::next(it) != all_idxs.end())
//             kernel_code += ", ";
//     }
//     kernel_code += ";\n\n\t";

//     for (auto &tensor : taco_tensors)
//     {
//         std::vector<std::string> format_str;
//         for (auto &idx : tensor.idxs)
//         {
//             format_str.push_back(std::to_string(id_val_map[idx]));         
//         }
//         kernel_code += (std::string("") + "Tensor<double> " + tensor.name + "(\"" + tensor.name + "\", {" + join(format_str) + "}, Format({" + tensor.storageFormat + "});\n\t");
//     }

//     kernel_code += "\n\t";
//     for(auto &tensor : taco_tensors)
//     {
//         if (tensor.name != 'A') // assuming A is always the output tensor
//         {
//             std::string for_loop = "";
//             std::vector<std::string> format_str;
//             for (size_t i =0; i < tensor.idxs.size(); i++)
//             {
//                 char idx = tensor.idxs[i];
//                 std::string loop_index = std::string(2, idx);
                
//                 // std::cout << for_loop << std::endl;
//                 format_str.push_back(loop_index);
//                 if (i == tensor.idxs.size() - 1)
//                 {
//                     for_loop += "for (int " + loop_index + " = 0; " + loop_index + " < " + std::to_string(id_val_map[idx]) + "; " + loop_index + "++) {\n" + std::string(8*(i+2), ' ');

//                     std::random_device rd;
//                     std::mt19937 gen(rd());
//                     std::uniform_real_distribution<> dist(0.0, 5.0);
//                     double randomValue = dist(gen);

//                     std::ostringstream oss;
//                     oss << std::fixed << std::setprecision(2) << randomValue;
//                     std::string strValue = oss.str();

//                     for_loop += "if insert_dist(gen) {\n" + std::string(8*(i+3), ' ');
//                     for_loop += std::string(1, tensor.name) + ".insert({" + join(format_str) + "}, " + strValue + ");\n" + std::string(8*(i+2), ' ') + "}";
//                 } else {
//                     for_loop += "for (int " + loop_index + " = 0; " + loop_index + " < " + std::to_string(id_val_map[idx]) + "; " + loop_index + "++)\n" + std::string(8*(i+2), ' ');
//                 }
//             }
//             kernel_code += for_loop + "\n";
//             kernel_code += "\t" + std::string(1, tensor.name) + ".pack();\n\n\t";
//         }
//     }

//     kernel_code += "\n}\n";
//     std::cout << kernel_code << std::endl;

//     return kernel_code;
// }

std::tuple<std::vector<tacoTensor>, std::string> random_valid_einsum(int numInputs = 2, int maxRank = 3) {
    static const std::string pool = "ijklmnopqrstu";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> rankDist(1, maxRank);
    std::uniform_int_distribution<> idxDist(0, pool.size() - 1);

    // Step 1: Generate tensors with unique indices
    std::vector<std::vector<char>> tensors(numInputs);
    std::map<char,int> idxCount;

    for (int t = 0; t < numInputs; ++t) {
        int rank = rankDist(gen);
        std::set<char> used;
        while ((int)used.size() < rank) {
            char c = pool[idxDist(gen)];
            if (used.count(c) == 0) {
                tensors[t].push_back(c);
                used.insert(c);
                idxCount[c]++;
            }
        }
    }

    // Step 2: Pick some indices as output indices
    std::vector<char> outputIdx;
    std::bernoulli_distribution isOutput(0.3); // ~30% of indices become output
    for (auto &p : idxCount) {
        if (isOutput(gen)) 
        {
            outputIdx.push_back(p.first);
        }
    }

    // Step 3: For all non-output indices that appear only once, duplicate in another tensor
    for (auto &p : idxCount) {
        char idx = p.first;
        if (std::find(outputIdx.begin(), outputIdx.end(), idx) != outputIdx.end()) continue;

        if (p.second == 1) {
            // Find the tensor that contains it
            int srcTensor = -1;
            for (int t = 0; t < numInputs; ++t) {
                if (std::find(tensors[t].begin(), tensors[t].end(), idx) != tensors[t].end()) {
                    srcTensor = t;
                    break;
                }
            }
            // Pick a different tensor to duplicate it
            int targetTensor = srcTensor;
            while (targetTensor == srcTensor) targetTensor = gen() % numInputs;
            tensors[targetTensor].push_back(idx);
            idxCount[idx]++; // now appears twice
        }
    }

    // Step 4: Build einsum string
    auto makeTensor = [&](char name, const std::vector<char>& idxs) {
        return std::string(1, name) + "(" + join(idxs) + ")";
    };

    auto makeTacoTensor = [&](std::vector<char> tensor_idx, char tensor_name, std::string taco_string_repr) {
        tacoTensor tensor;
        tensor.idxs = tensor_idx;
        tensor.name = tensor_name;
        tensor.str_repr = taco_string_repr;
        std::vector<std::string> format_str;
        for (size_t i = 0; i < tensor_idx.size(); i++)
        {
            TensorFormat format = randomFormat(gen);
            switch (format)
            {
            case TensorFormat::tDense:
                tensor.storageFormat.push_back("Dense");
                break;
            case TensorFormat::tSparse:
                tensor.storageFormat.push_back("Sparse");
                break;
            default:
                break;
            }
        }
        // tensor.storageFormat = join(format_str);
        return tensor;
    };

    std::vector<tacoTensor> taco_tensors = {};
    std::string lhs = "A(" + join(outputIdx) + ")";
    taco_tensors.push_back(makeTacoTensor(outputIdx, 'A', lhs));
    std::string rhs;
    for (int i = 0; i < numInputs; ++i) {
        if (i > 0) rhs += " * ";

        std::string tensor_str = makeTensor('B' + i, tensors[i]);
        rhs += tensor_str;

        taco_tensors.push_back(makeTacoTensor(tensors[i], ('B'+i), tensor_str));
    }

    return {taco_tensors, rhs};
}