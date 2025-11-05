#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>

#include "tensure/formats.hpp"
#include "tensure/utils.hpp"

using namespace std;

tuple<vector<tsTensor>, string> generate_random_einsum(int numInputs = 2, int maxRank = 3);