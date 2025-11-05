#include "tensure/utils.hpp"
/**
 * Utility: Join strings with commas as the separator.
 * @param idxs Vector of strings
 * @return Comma-separated string
 */
string join(const vector<string>& idxs) {
    string s;
    for (size_t i = 0; i < idxs.size(); ++i) {
        s += idxs[i];
        if (i + 1 < idxs.size()) s += ",";
    }
    return s;
}
