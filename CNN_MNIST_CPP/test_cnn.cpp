#include <gtest/gtest.h>
#include "ConvolutionLayer.h"
#include "MaxpoolingLayer.h"

// Define the CNN architecture
constexpr int input_channels = 1; // MNIST images are grayscale, so only one channel
constexpr int kernel_size = 3;
constexpr int output_channels = 16; // Number of output channels for convolution layer
constexpr int max_pooling_decimation = 2; // Decimation rate for max pooling
constexpr int fc_input_units = 16 * 13 * 13; // Input units for the fully connected layer
constexpr int fc_output_units = 10; // Output units for the fully connected layer
constexpr double alpha = 0.01; //learning rate

// Load pre-trained weights, biases, and kernels
image2D flattened_kernels = read_csv("kernels_training.csv");
image3D kernels = reshape_to_3d(flattened_kernels, output_channels, kernel_size);

TEST(VariableSizeTest, TestConvOutputSize) {

    constexpr int num_test_images_to_select = 1;

    std::string test_images_file = "t10k-images-idx3-ubyte.gz";
    std::string test_labels_file = "t10k-labels-idx1-ubyte.gz";

    image3D X_test = read_mnist_images(test_images_file);
    image1D Y_test = read_mnist_labels(test_labels_file);

    if (num_test_images_to_select < X_test.size()) {
        X_test.resize(num_test_images_to_select);
        Y_test.resize(num_test_images_to_select);
    }

    ConvolutionLayer conv1(output_channels, kernel_size, alpha, kernels);
    image4D batch_test = reshape_to_batch(X_test);

    // Calculate the expected size for conv_output based on your input data and layer configuration
    constexpr int expected_conv_output_h = 26;
    constexpr int expected_conv_output_w = 26;
    constexpr int expected_conv_output_c = 16;

    // Calculate the size of conv_output
    image3D conv_output = conv1.forward_prop(batch_test[0]);

    // Check the size of conv_output using Google Test assertions
    ASSERT_EQ(conv_output.size(), expected_conv_output_h);
    ASSERT_EQ(conv_output[0].size(), expected_conv_output_w);
    ASSERT_EQ(conv_output[0][0].size(), expected_conv_output_c);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}