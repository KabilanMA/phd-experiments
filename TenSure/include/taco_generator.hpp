#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>
// #include "taco.h"
#include <fstream>

enum TensorFormat {
    tSparse=0,
    tDense=1
};

typedef struct tacoTensor
{
    char name;
    std::string str_repr;
    std::vector<char> idxs;
    std::vector<std::string> storageFormat;
} tacoTensor;


std::tuple<std::vector<tacoTensor>, std::string> random_valid_einsum(int numInputs, int maxRank);
std::ostream& operator<<(std::ostream& os, const tacoTensor& tensor);
// std::set<char> find_idxs(std::vector<tacoTensor> taco_tensors);
std::map<char, int> map_id_to_val(std::set<char> idxs);
std::string& generate_taco_kernel(std::vector<tacoTensor> taco_tensors, std::string einsum_rhs);
bool generate_store_tensor_data(const std::vector<tacoTensor>& taco_tensors, std::string file_name_suffix, std::string location);
