#include <iostream>
#include <vector>
#include <cmath>
#include "ConvolutionLayer.cu"
#include "Image.h"
#include "Functions.h"
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <iomanip>

int main() {
	
    int blocks = 2 ;

    // Create a 3x3 matrix
    image3D image(3, image2D(3, image1D(1, 0.0)));

    // Initialize the matrix with numbers from 1 to 9
    double num = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0;k < 1; k++) {
                image[i][j][k] = num;
                num++;
            }
        }
    }

//    image3D conv_output = conv_forward_prop(image, kernel, blocks);
    // Define the kernel size and number of channels
    int kernel_size = 2;
    int num_channels = 3;

    // Create a 3D vector to represent the kernel
    std::vector<std::vector<std::vector<double>>> kernel(num_channels,
                                                         std::vector<std::vector<double>>(kernel_size,
                                                                                           std::vector<double>(kernel_size, 0.0)));

    // Initialize the kernel with some values (you can set your desired values here)
    double nums = 0.0;
    for (int c = 0; c < num_channels; ++c) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                // Set your values here, for example:
                kernel[c][i][j] = nums;
		nums ++;
            }
        }
    }

    // Print the kernel
    for (int c = 0; c < num_channels; ++c) {
        std::cout << "Channel " << c << ":\n";
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                std::cout << kernel[c][i][j] << " ";
            }
            std::cout << "\n"; // Newline after each row
        }
    }


	image3D conv_output = conv_forward_prop(image, kernel, blocks);
    return 0;
}
