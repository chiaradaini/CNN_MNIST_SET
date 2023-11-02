#include <iostream>
#include <vector>
#include <cmath>
#include "ConvolutionLayer.h"
#include "Image.h"
#include "Functions.h"
#include <vector>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <iomanip>

// Define the CNN architecture
constexpr int input_channels = 1; // MNIST images are grayscale, so only one channel
constexpr int kernel_size = 2;
constexpr int output_channels = 3; // Number of output channels for convolution layer
constexpr double alpha = 0.01; //learning rate


int main() {

    int taglia = 10;

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


    ConvolutionLayer conv1(output_channels, kernel_size, alpha, kernel);
    image3D conv_output = conv1.forward_prop(image);
    print_kernels(conv_output);

    return 0;
}
