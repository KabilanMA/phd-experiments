#include "tensure/random_gen.hpp"

// /**
//  * Utility: Overload the output stream operator for tacoTensor.
//  * @param os Output stream
//  * @param tensor tacoTensor to print
//  * @return Output stream
//  */
// std::ostream& operator<<(std::ostream& os, const tsTensor& tensor) 
// {
//     os << tensor.str() ;
//     return os;
// }

tuple<vector<tsTensor>, string> generate_random_einsum(int numInputs, int maxRank) {
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

    auto makeTsTensor = [&](std::vector<string> tensor_idx, char tensor_name, std::string taco_string_repr) {
        tsTensor tensor;
        tensor.axes = tensor_idx;
        tensor.name = string(1, tensor_name);
        tensor.str_repr = taco_string_repr;
        vector<string> format_str;
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

    vector<tsTensor> tstensors = {};
    string lhs = "A(" + join(outputIdx) + ")";
    tstensors.push_back(makeTsTensor(outputIdx, 'A', lhs));
    std::string rhs;
    for (int i = 0; i < numInputs; ++i) {
        if (i > 0) rhs += " * ";

        std::string tensor_str = makeTensor('B' + i, tensors[i]);
        rhs += tensor_str;

        taco_tensors.push_back(makeTacoTensor(tensors[i], ('B'+i), tensor_str));
    }

    return {taco_tensors, rhs};
}