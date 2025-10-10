FROM ubuntu:22.04

# Install build tools and dependencies
RUN apt-get update && \
    apt-get install -y build-essential cmake vim git wget curl make && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone and build TACO
RUN git clone --recursive https://github.com/tensor-compiler/taco.git /workspace/tools/taco && \
    mkdir -p /workspace/tools/taco/build && \
    cd /workspace/tools/taco/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Add TACO to library path
ENV LD_LIBRARY_PATH=/workspace/tools/taco/build/lib:$LD_LIBRARY_PATH

# Default entrypoint: execute any given command
ENTRYPOINT ["/bin/bash", "-c"]
