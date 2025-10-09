#include <formats.hpp>
#include <fstream>
#include <filesystem>
#include "taco.h"
#include "taco_kernel.hpp"
#include "rand_gen.hpp"
#include "raw_kernel.hpp"
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <sys/resource.h>


using namespace taco;

int calculate_nnz_per_row(int value, float percentage)
{
    int result = (int)ceil(value*percentage);
    return result > value ? value : result;
}

std::string create_results_file(const std::string& filename, const std::string& header, const std::string& experiment)
{
    std::filesystem::path folder_path = "./results/" + experiment;

    // Create directories if they don't exist
    std::error_code ec; // prevent exceptions
    if (!std::filesystem::exists(folder_path)) {
        if (!std::filesystem::create_directories(folder_path, ec)) {
            std::cerr << "Failed to create folder: " << folder_path 
                      << " (" << ec.message() << ")" << std::endl;
            return "";
        }
    }

    // Create the file
    std::ofstream file(folder_path / filename);
    if (!file.is_open()) {
        std::cerr << "Failed to create file: " << filename << std::endl;
        return "";
    }

    file << header << std::endl;
    file.close();
    return (folder_path / filename).string();
}

bool save_results_to_csv(const std::string& filename, int dim, int nnz_per_row, float sparsity, double taco_time, double raw_kernel_time)
{
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    file << dim << "," << nnz_per_row << "," << sparsity << "," << taco_time << "," << raw_kernel_time << "\n";
    file.close();
    return true;
}

template<typename... Args>
bool save_to_csv(const std::string& filename, Args&&... args)
{
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // Fold expression to unpack arguments and separate them with commas
    ((file << args << ","), ...);

    // Replace last comma with newline
    file.seekp(-1, std::ios_base::cur);
    file << "\n";

    file.close();
    return true;
}


// Helper: run kernel safely in a child process
double run_kernel_safe(std::function<double()> kernel_func,
                       int timeout_seconds,
                       size_t max_memory_bytes,
                       bool& success)
{
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        perror("pipe failed");
        success = false;
        return -1.0;
    }

    pid_t pid = fork();
    if (pid == -1) {
        perror("fork failed");
        success = false;
        return -1.0;
    }

    if (pid == 0) { // Child
        close(pipefd[0]); // close read end

        // Set memory limit
        struct rlimit rl;
        rl.rlim_cur = rl.rlim_max = max_memory_bytes;
        setrlimit(RLIMIT_AS, &rl);

        // Run kernel
        double result = kernel_func();

        // Send result to parent
        write(pipefd[1], &result, sizeof(result));
        close(pipefd[1]);
        exit(0);
    } else { // Parent
        close(pipefd[1]); // close write end

        int waited = 0;
        int status;
        while (waited < timeout_seconds) {
            pid_t ret = waitpid(pid, &status, WNOHANG);
            if (ret != 0) break; // child finished
            std::this_thread::sleep_for(std::chrono::seconds(1));
            waited++;
        }

        if (waited >= timeout_seconds) {
            std::cerr << "Kernel timed out! Killing process " << pid << "\n";
            kill(pid, SIGKILL);
            success = false;
            return -1.0;
        }

        double result = -1.0;
        ssize_t n = read(pipefd[0], &result, sizeof(result));
        close(pipefd[0]);

        if (n != sizeof(result) || !WIFEXITED(status)) {
            std::cerr << "Kernel failed (memory limit or crash)\n";
            success = false;
            return -1.0;
        }

        success = true;
        return result;
    }
}

/**
 * Sparsity == percentage of nnz per row
 */
void experiment2(float sparsity_incrementor)
{
    float sparsity = sparsity_incrementor;
    const int timeout_seconds = 10; // max runtime per kernel
    const size_t max_memory_bytes = 20L * 1024 * 1024 * 1024; // 20 GiB per kernel

    while (sparsity < 1)
    {
        // multiple runs to average out
        for (size_t run = 0; run < 1; run++)
        {
            std::string results_file = create_results_file(("results_" + std::to_string(run) + ".csv"), "Dimension,NNZ_per_row,Sparsity,TACO_Time,Raw_Time", "experiment2");
            // for different matrix sizes
            for (int dim = 3; dim < 5000; dim++)
            {
                int nnz_per_row = calculate_nnz_per_row(dim, sparsity);
                COOMatrix B = generate_synthetic_matrix(dim, dim, nnz_per_row);
                COOMatrix C = generate_synthetic_matrix(dim, dim, nnz_per_row);
                Tensor<double> workspace;

                // bool success = false;
                // double taco_time = run_kernel_safe([&]() {
                //     return taco_kernel_1_1(B, C, workspace);
                // }, timeout_seconds, max_memory_bytes, success);

                // if (!success) {
                //     std::cout << "Skipping this run due to taco kernel failure\n";
                //     continue;
                // }

                // double raw_kernel_time = run_kernel_safe([&]() {
                //     return raw_kernel_1_1(B, C);
                // }, timeout_seconds, max_memory_bytes, success);
                double raw_kernel_time = raw_kernel_1_1(B, C);
                double taco_time = taco_kernel_1_1(B, C, workspace);


                freeCOOMatrix(&B);
                freeCOOMatrix(&C);
                
                // if (!success) {
                //     std::cout << "Skipping this run due to raw kernel failure\n";
                //     continue;
                // }

                // Save results to CSV
                save_to_csv(results_file, dim, nnz_per_row, sparsity, taco_time, raw_kernel_time);
                
                std::cout << "Dim: " << dim << ", NNZ/Row: " << nnz_per_row 
                         << ", Sparsity: " << sparsity 
                         << ", TACO: " << taco_time 
                         << ", Raw: " << raw_kernel_time << std::endl;
                std::cout << "=================================================" << std::endl;
            }
        }
        sparsity += sparsity_incrementor;
    }
}

int main(int argc, char *argv[])
{
    experiment2(0.05);
    // create_results_file("results.txt", "Dimension,NNZ_per_row,Sparsity,TACO_Time,Raw_Time", "experiment2");
    // COOMatrix B = generate_synthetic_matrix(3, 3, 2);
    // COOMatrix C = generate_synthetic_matrix(3, 3, 2);
    // Tensor<double> workspace;
    // taco_kernel_1_1(B, C, workspace);
    // std::cout << workspace << std::endl;
    return 0;
}