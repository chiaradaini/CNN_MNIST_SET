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
constexpr int max_pooling_decimation = 2; // Decimation rate for max pooling
constexpr int fc_input_units = 16 * 13 * 13; // Input units for the fully connected layer
constexpr int fc_output_units = 10; // Output units for the fully connected layer
constexpr double alpha = 0.01; //learning rate

int main() {

    constexpr int num_test_images_to_select = 1;

    // Specify the file paths of the MNIST dataset
    std::string test_images_file = "t10k-images-idx3-ubyte.gz";
    std::string test_labels_file = "t10k-labels-idx1-ubyte.gz";

    // Read the MNIST images
    image3D X_test = read_mnist_images(test_images_file);
    image1D Y_test = read_mnist_labels(test_labels_file);

    // Select the specified number of images from the test set
    if (num_test_images_to_select < X_test.size()) {
        X_test.resize(num_test_images_to_select);
        Y_test.resize(num_test_images_to_select);
    }

    // Load pre-trained weights, biases, and kernels
    image2D flattened_kernels = read_csv("kernels_training.csv");
    image2D weights = read_csv("weights_training.csv");
    image1D biases = read_csv_image1D("biases_training.csv");
    image3D kernels = reshape_to_3d(flattened_kernels, output_channels, kernel_size);

    // Define the Layers
    ConvolutionLayer conv1(output_channels, kernel_size, alpha, kernels);
    MaxPoolingLayer max_pooling(max_pooling_decimation);
    FCLayer fc_layer(fc_input_units, fc_output_units, alpha, weights, biases);

    image4D batch_test = reshape_to_batch(X_test);

    // Evaluate the CNN on the test set
    int correct_predictions = 0;
    for (size_t i = 0; i < batch_test.size(); ++i) {
        image3D conv_output = conv1.forward_prop(batch_test[i]);
        image3D max_pool_output = max_pooling.forward_prop(conv_output);
        image1D flatten_output = convert_to_flattened_input(max_pool_output);
        image1D fc_output = fc_layer.forward_prop(flatten_output);

        // Find the predicted label
        int predicted_label = static_cast<int>(std::distance(fc_output.begin(), std::max_element(fc_output.begin(), fc_output.end())));
        if (predicted_label == Y_test[i]) {
            correct_predictions++;
        }
    }

    double accuracy = static_cast<double>(correct_predictions) / X_test.size() * 100.0;
    std::cout << "Test accuracy: " << accuracy << "%" << std::endl;

    return 0;
}