#include <iostream>
#include <vector>
#include <cmath>
#include "ConvolutionLayer.cu"
#include "Image.h"
#include "Functions.h"
#include <vector>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <iomanip>

int main() {

    int taglia = 3;

    image3D image(taglia, image2D(taglia, image1D(1, 0.0)));

    int num = 0;
    for (int i = 0; i < taglia; i++) {
        for (int j = 0; j < taglia; j++) {
            image[i][j][0] = num;
            num++;
        }
    }

    // // Print the matrix
    // std::cout << "Image:" << std::endl;
    // for (int i = 0; i < taglia; i++) {
    //     for (int j = 0; j < taglia; j++) {
    //         std::cout << image[i][j][0] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Define the kernel size and number of channels
    int kernel_size = 2;
    int num_channels = 3;
    std::vector<float> executionTimes;

    // Create a 3D vector to represent the kernel
    std::vector<std::vector<std::vector<double>>> kernel(num_channels,
                                                         std::vector<std::vector<double>>(kernel_size,
                                                           std::vector<double>(kernel_size, 0.0)));

    // Initialize the kernel with some values
    double nums = 0.0;
    for (int c = 0; c < num_channels; ++c) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                kernel[c][i][j] = nums;
                nums++;
            }
        }
    }

    // //Print the kernel
    // std::cout << "Kernel:" << std::endl;
    // for (int c = 0; c < num_channels; ++c) {
    //     for (int i = 0; i < kernel_size; ++i) {
    //         for (int j = 0; j < kernel_size; ++j) {
    //             std::cout << kernel[c][i][j] << " ";
    //         }
    //         std::cout << "\n"; // Newline after each row
    //     }
    // }

    // Prepare the kernel for GPU processing
    image1D flattened_kernels = convert_to_flattened_input(kernel);
    double* dev_flattened_kernels;
    cudaMalloc((void**)&dev_flattened_kernels, flattened_kernels.size() * sizeof(double));
    cudaMemcpy(dev_flattened_kernels, flattened_kernels.data(), flattened_kernels.size() * sizeof(double), cudaMemcpyHostToDevice);

    int numBlocks =1;
    int numThreads = 32;

    ConvolutionResult result = conv_forward_prop(image, dev_flattened_kernels, kernel_size, num_channels, numBlocks, numThreads);
    image3D conv_output = result.conv_output;

    print_kernels(result.conv_output);
    float milliseconds = result.milliseconds;
    executionTimes.push_back(milliseconds);

    cudaFree(dev_flattened_kernels);

    return 0;
}
