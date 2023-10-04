#include <iostream>
#include <vector>
#include <cmath>
#include "ConvolutionLayer.h"
#include "MaxPoolingLayer.h"
#include "FCLayer.h"
#include "Image.h"
#include "Functions.h"
#include "BackPropResults.h"
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <iomanip>

// Define the CNN architecture
constexpr int input_channels = 1; // MNIST images are grayscale, so only one channel
constexpr int kernel_size = 3;
constexpr int output_channels = 16; // Number of output channels for convolution layer
constexpr int m_decimation = 2; // Decimation rate for max pooling
constexpr int fc_input_units = 16 * 13 * 13; // Input units for the fully connected layer
constexpr int fc_output_units = 10; // Output units for the fully connected layer
constexpr double alpha = 0.01; //learning rate

int main() {

    constexpr int num_test_images_to_select = 1;

    std::string test_images_file = "t10k-images-idx3-ubyte.gz";
    std::string test_labels_file = "t10k-labels-idx1-ubyte.gz";

    image3D X_test = read_mnist_images(test_images_file);
    image1D Y_test = read_mnist_labels(test_labels_file);

    if (num_test_images_to_select < X_test.size()) {
        X_test.resize(num_test_images_to_select);
        Y_test.resize(num_test_images_to_select);
    }

    // Load pre-trained weights, biases, and kernels
    image2D flattened_kernels = read_csv("kernels_training.csv");
    image2D weights = read_csv("weights_training.csv");
    image1D biases = read_csv_image1D("biases_training.csv");
    image3D kernels = reshape_to_3d(flattened_kernels, output_channels, kernel_size);

    ConvolutionLayer conv1(output_channels, kernel_size, alpha, kernels);
    MaxPoolingLayer max_pooling(m_decimation);
    image4D batch_test = reshape_to_batch(X_test);

    image3D conv_output = conv1.forward_prop(batch_test[0]);
    int output_h = conv_output.size() / m_decimation;
    int output_w = conv_output[0].size() / m_decimation;

    image1D flattened_expected_maxpool_output = read_csv_image1D_conv("max_pool_output.csv");
    image3D expected_maxpool_output = reshape_to_3d_conv(flattened_expected_maxpool_output, output_h, output_w, output_channels);

    print_kernels(expected_maxpool_output);
    image3D maxpool_output = max_pooling.forward_prop(conv_output);

    //Check if conv_output is approximately equal to expected_conv_output element-wise within a tolerance
    double tolerance = 1e-4; // Adjust the tolerance as needed
    for (size_t c = 0; c < maxpool_output[0][0].size(); ++c) {
        for (size_t h = 0; h < maxpool_output[0].size(); ++h) {
            for (size_t w = 0; w < maxpool_output.size(); ++w) {
                if ( maxpool_output[h][w][c] != expected_maxpool_output[h][w][c]){
                    std::cout << "h: " << h << std::endl;
                    std::cout << "w: " << w << std::endl;
                    std::cout << "c: " << c << std::endl;
                    std::cout << "maxpool_output: " << maxpool_output[h][w][c] << std::endl;
                    std::cout << "expected_maxpool_output: " << expected_maxpool_output[h][w][c] << std::endl;
                }
            }
        }
    }


    return 0;
}
