#!/usr/bin/env bash
set -e  # Exit on first error
set -o pipefail

# Root project directory (one level above where script is located)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TACO_DIR="$ROOT_DIR/external/taco"
BUILD_DIR="$TACO_DIR/build"

# Colorful log output
log() {
    echo -e "\033[1;32m[+] $1\033[0m"
}

err() {
    echo -e "\033[1;31m[!] $1\033[0m"
}

# ---------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------
if ! command -v cmake &>/dev/null; then
    err "CMake not found. Please install CMake >= 3.4.0"
    exit 1
fi

if ! command -v g++ &>/dev/null && ! command -v clang++ &>/dev/null; then
    err "No C++ compiler (g++ or clang++) found. Please install one."
    exit 1
fi

# ---------------------------------------------------------------------
# Initialize submodules (with nested ones)
# ---------------------------------------------------------------------
if [ ! -d "$TACO_DIR" ]; then
    log "TACO not found in external/. Clone it first."
    log "Example: git submodule add https://github.com/tensor-compiler/taco.git external/taco"
    exit 1
fi

log "Initializing and updating submodules..."
pushd "$TACO_DIR" > /dev/null
git submodule update --init --recursive --depth 1
popd > /dev/null

# ---------------------------------------------------------------------
# Configure and build
# ---------------------------------------------------------------------
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

log "Configuring TACO build..."
pushd "$BUILD_DIR" > /dev/null

# Allow user to override compiler by exporting CC/CXX beforehand
cmake -DCMAKE_BUILD_TYPE=Release ..

log "Building TACO (using $(nproc) cores)..."
make -j"$(nproc)"

popd > /dev/null

log "✅ TACO built successfully!"
log "Library output: $BUILD_DIR/libtaco.a"
