#!/bin/bash
set -e

# Usage: ./run_experiment.sh <sparsity> <exp_id>
SPARSITY=${1:-1}
EXP_ID=${2:-1}

echo "🚀 Starting experiment with sparsity=${SPARSITY} and exp_id=${EXP_ID}"

# Compile the code
make taco FILE=./src/main.cpp

# Run the program
./src/main.out "$SPARSITY" "$EXP_ID"

echo "✅ Experiment finished successfully."