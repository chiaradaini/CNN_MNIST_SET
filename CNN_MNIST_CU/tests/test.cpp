#include <iostream>
#include <vector>
#include <cmath>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <iomanip>
#include <cassert>

#include "../inc/Convolution.h"
#include "../inc/memory_allocation.h"
#include "../inc/Image.h"
#include "../inc/Functions.h"

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

    std::cout << "Image:" << std::endl;
    for (int i = 0; i < taglia; i++) {
        for (int j = 0; j < taglia; j++) {
            std::cout << image[i][j][0] << " ";
        }
        std::cout << std::endl;
    }

    int kernel_size = 2;
    int num_channels = 3;
    std::vector<float> executionTimes;

    image3D kernel(num_channels, image2D(kernel_size, image1D(kernel_size, 0.0)));

    double nums = 0.0;
    for (int c = 0; c < num_channels; ++c) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                kernel[c][i][j] = nums;
                nums++;
            }
        }
    }

    std::cout << "Kernel:" << std::endl;
    for (int c = 0; c < num_channels; ++c) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                std::cout << kernel[c][i][j] << " ";
            }
            std::cout << "\n";
        }
    }

    image1D flattened_kernels = convert_to_flattened_input(kernel);
    double* dev_flattened_kernels = nullptr;
    AllocateMemory(&dev_flattened_kernels, flattened_kernels.data(), flattened_kernels.size());
    CopyMemoryToDevice(&dev_flattened_kernels, flattened_kernels.data(), flattened_kernels.size());

    double* host_flattened_kernels = new double[flattened_kernels.size()];
    CopyMemoryToHost(host_flattened_kernels, &dev_flattened_kernels, flattened_kernels.size());

    std::cout << "Kernel from GPU memory:" << std::endl;
    for (size_t i = 0; i < flattened_kernels.size(); ++i) {
        std::cout << host_flattened_kernels[i] << " ";
    }
    std::cout << std::endl;

    delete[] host_flattened_kernels;

    int numBlocks =1;
    int numThreads = 32;
    int granularity = 2;

    ConvolutionResult result = conv_forward_prop(image, dev_flattened_kernels, kernel_size, num_channels, numBlocks, numThreads, granularity);
    image3D conv_output = result.conv_output;

    float milliseconds = result.milliseconds;
    executionTimes.push_back(milliseconds);

    FreeMemory(dev_flattened_kernels);

    return 0;
}
