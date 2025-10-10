### Experimental Setup Guides

Running the experiments might exhaust the CPU cycles and cumulative memory to slow-down the system process and can lead to system crash without graceful termination. Therefore we will create a sandbox envionment to run the experiments with CPU and RAM constraints.

1. Build the docker image

```bash
docker build -t spexp .
```

2. (Option 1) Build the experiments with the TACO build in the container and then run another container to use that build to run the experiments.
    One time build and then re-run the program.

Build

```bash
docker run --rm -v /home/kabilan/Desktop/phd-experiments/tensor-contraction:/workspace/tensor-contraction spexp "cd /workspace/tensor-contraction && make taco FILE=./src/main.cpp"
```

Run

```bash
docker run --rm --cpus="2" --memory="4g" -v /home/kabilan/Desktop/phd-experiments/tensor-contraction:/workspace/tensor-contraction spexp "cd /workspace/tensor-contraction && ./src/main.out"
```

3. (Option 2) Build and Run every single time.

```bash
docker run --rm -v /home/kabilan/Desktop/phd-experiments/tensor-contraction:/workspace/tensor-contraction spexp "cd /workspace/tensor-contraction && make taco FILE=./src/main.cpp && ./src/main.out"
```