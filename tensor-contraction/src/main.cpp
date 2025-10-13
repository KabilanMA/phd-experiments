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


/**
 * Sparsity == percentage of nnz per row
 */
void experiment2(float sparsity_starter)
{
    float sparsity = sparsity_starter;

    while (sparsity < 1)
    {
        // multiple runs to average out
        for (size_t run = 0; run < 1; run++)
        {
            std::string results_file = create_results_file(("results_raw_" + std::to_string(run) + "_" + std::to_string((int)(sparsity*100)) + ".csv"), "Dimension,NNZ_per_row,Sparsity,Raw_Time", "experiment2");
            // for different matrix sizes
            for (int dim = 3; dim < 2000; dim++)
            {
                int nnz_per_row = calculate_nnz_per_row(dim, sparsity);
                COOMatrix B = generate_synthetic_matrix(dim, dim, nnz_per_row);
                COOMatrix C = generate_synthetic_matrix(dim, dim, nnz_per_row);
                Tensor<double> workspace;

                double raw_kernel_time = raw_kernel_1_1(B, C);
                // double taco_time = taco_kernel_1_1(B, C, workspace);

                workspace = Tensor<double>();
                freeCOOMatrix(&B);
                freeCOOMatrix(&C);

                // Save results to CSV
                save_to_csv(results_file, dim, nnz_per_row, sparsity, raw_kernel_time);
                
                std::cout << "Dim: " << dim << ", NNZ/Row: " << nnz_per_row 
                        << ", Sparsity: " << sparsity
                        //  << ", TACO: " << taco_time 
                         << ", Raw Time: " << raw_kernel_time << std::endl;
                std::cout << "=================================================" << std::endl;
            }
        }
        sparsity += 0.05;
    }
}

/**
 * Sparsity == percentage of nnz per row
 */
void experiment3(float sparsity_starter)
{
    float sparsity = sparsity_starter;

    while (sparsity < 1)
    {
        // multiple runs to average out
        for (size_t run = 0; run < 1; run++)
        {
            std::string results_file = create_results_file(("results_taco_" + std::to_string(run) + "_" + std::to_string((int)(sparsity*100)) + ".csv"), "Dimension,NNZ_per_row,Sparsity,TACO_Time", "experiment3");
            // for different matrix sizes
            for (int dim = 3; dim < 1000; dim++)
            {
                int nnz_per_row = calculate_nnz_per_row(dim, sparsity);
                COOMatrix B = generate_synthetic_matrix(dim, dim, nnz_per_row);
                COOMatrix C = generate_synthetic_matrix(dim, dim, nnz_per_row);
                Tensor<double> workspace;

                // double raw_kernel_time = raw_kernel_1_1(B, C);
                double taco_time = taco_kernel_1_1(B, C, workspace);

                workspace = Tensor<double>();
                freeCOOMatrix(&B);
                freeCOOMatrix(&C);

                // Save results to CSV
                save_to_csv(results_file, dim, nnz_per_row, sparsity, taco_time);
                
                std::cout << "Dim: " << dim << ", NNZ/Row: " << nnz_per_row 
                        << ", Sparsity: " << sparsity
                         << ", TACO: " << taco_time  << std::endl;
                        //  << ", Raw Time: " << raw_kernel_time << std::endl;
                std::cout << "=================================================" << std::endl;
            }
        }
        std::cout << "====================Completed " << sparsity << "====================" << std::endl;
        sparsity += 0.05;
    }
}

void experiment4(float sparsity_starter)
{
    float sparsity = sparsity_starter;

    while (sparsity < 1)
    {
        std::string results_file = create_results_file(("results_taco_" + std::to_string((int)(sparsity*100)) + ".csv"), "Dimension,NNZ_per_row,Sparsity,Time", "experiment4");
        for (int dim = 3; dim < 1001; dim++)
        {
            int nnz_per_row = calculate_nnz_per_row(dim, sparsity);
            COOMatrix B = generate_synthetic_matrix(dim, dim, nnz_per_row);
            COOMatrix C = generate_synthetic_matrix(dim, dim, nnz_per_row);
            Tensor<double> workspace;

            double taco_time = taco_kernel_2_1(B, C, workspace);

            workspace = Tensor<double>();
            freeCOOMatrix(&B);
            freeCOOMatrix(&C);

            // Save results to CSV
            save_to_csv(results_file, dim, nnz_per_row, sparsity, taco_time);

            std::cout << "Dim: " << dim << ", NNZ/Row: " << nnz_per_row 
                    << ", Sparsity: " << sparsity
                    << ", TACO: " << taco_time  << std::endl;
                std::cout << "=================================================" << std::endl;
        }

        sparsity += 0.05;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s double<starting_sparsity_value> <int_value>(0-TACO, 1-RAW)\n", argv[0]);
        return 1;
    }

    double sparsity_starter = std::stof(argv[1]);
    int experiment_type = std::stoi(argv[2]);
    switch (experiment_type)
    {
    case 0:
        /* Call the TACO experiment3 - TACO Kernel*/
        experiment4(sparsity_starter);
        break;
    default:
        /* Call the experiment2 - Raw Kernel */
        experiment4(sparsity_starter);
        break;
    }
    // create_results_file("results.txt", "Dimension,NNZ_per_row,Sparsity,TACO_Time,Raw_Time", "experiment2");
    // COOMatrix B = generate_synthetic_matrix(3, 3, 2);
    // COOMatrix C = generate_synthetic_matrix(3, 3, 2);
    // Tensor<double> workspace;
    // taco_kernel_1_1(B, C, workspace);
    // std::cout << workspace << std::endl;
    return 0;
}