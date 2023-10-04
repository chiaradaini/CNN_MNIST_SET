#include <gtest/gtest.h>
#include "MaxPoolingLayer.h"
#include "ConvolutionLayer.h"
#include "FCLayer.h"

// Define the CNN architecture
constexpr int input_channels = 1; // MNIST images are grayscale, so only one channel
constexpr int kernel_size = 3;
constexpr int output_channels = 16; // Number of output channels for convolution layer
constexpr int m_decimation = 2; // Decimation rate for max pooling
constexpr int fc_input_units = 16 * 13 * 13; // Input units for the fully connected layer
constexpr int fc_output_units = 10; // Output units for the fully connected layer
constexpr double alpha = 0.01; //learning rate

// Load pre-trained weights, biases, and kernels
image2D flattened_kernels = read_csv("kernels_training.csv");
image2D weights = read_csv("weights_training.csv");
image1D biases = read_csv_image1D("biases_training.csv");
image3D kernels = reshape_to_3d(flattened_kernels, output_channels, kernel_size);

TEST(VariableSizeTest, TestFCOutputSize) {

    constexpr int num_test_images_to_select = 1;

    std::string test_images_file = "t10k-images-idx3-ubyte.gz";

    image3D X_test = read_mnist_images(test_images_file);

    if (num_test_images_to_select < X_test.size()) {
        X_test.resize(num_test_images_to_select);
    }

    ConvolutionLayer conv1(output_channels, kernel_size, alpha, kernels);
    MaxPoolingLayer max_pooling(m_decimation);
    FCLayer fc_layer(fc_input_units, fc_output_units, alpha, weights, biases);
    image4D batch_test = reshape_to_batch(X_test);

    image3D conv_output = conv1.forward_prop(batch_test[0]);
    image3D maxpool_output = max_pooling.forward_prop(conv_output);
    image1D flatten_output = convert_to_flattened_input(maxpool_output);
    image1D fc_output = fc_layer.forward_prop(flatten_output);
    image1D expected_FC_output = read_csv_image1D_conv("fc_output.csv");

    // Calculate the expected size for conv_output based on your input data and layer configuration
    int expected_FC_output_size = expected_FC_output.size();

    // Check the size of conv_output using Google Test assertions
    ASSERT_EQ(fc_output.size(), expected_FC_output_size);
}

TEST(ValuesConvOutput, TestFCOutputValues) {

    constexpr int num_test_images_to_select = 1;

    std::string test_images_file = "t10k-images-idx3-ubyte.gz";

    image3D X_test = read_mnist_images(test_images_file);

    if (num_test_images_to_select < X_test.size()) {
        X_test.resize(num_test_images_to_select);
    }

    ConvolutionLayer conv1(output_channels, kernel_size, alpha, kernels);
    MaxPoolingLayer max_pooling(m_decimation);
    FCLayer fc_layer(fc_input_units, fc_output_units, alpha, weights, biases);
    image4D batch_test = reshape_to_batch(X_test);

    image3D conv_output = conv1.forward_prop(batch_test[0]);
    image3D maxpool_output = max_pooling.forward_prop(conv_output);
    image1D flatten_output = convert_to_flattened_input(maxpool_output);
    image1D fc_output = fc_layer.forward_prop(flatten_output);
    image1D expected_FC_output = read_csv_image1D_conv("fc_output.csv");

    // Check if conv_output is approximately equal to expected_conv_output element-wise within a tolerance
    double tolerance = 1e-4; // Adjust the tolerance as needed
    for (size_t h = 0; h < expected_FC_output.size(); ++h) {
        ASSERT_NEAR(fc_output[h], expected_FC_output[h], tolerance);
    }

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}