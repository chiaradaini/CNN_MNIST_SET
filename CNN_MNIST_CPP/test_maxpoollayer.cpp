#include <gtest/gtest.h>
#include "MaxPoolingLayer.h"
#include "ConvolutionLayer.h"

// Define the CNN architecture
constexpr int kernel_size = 3;
constexpr int output_channels = 16; // Number of output channels for convolution layer
constexpr double alpha = 0.01; //learning rate
constexpr int m_decimation = 2; // Decimation rate for max pooling

// Load pre-trained weights, biases, and kernels
image2D flattened_kernels = read_csv("kernels_training.csv");
image3D kernels = reshape_to_3d(flattened_kernels, output_channels, kernel_size);

TEST(VariableSizeTest, TestMaxPoolOutputSize) {

    constexpr int num_test_images_to_select = 1;

    std::string test_images_file = "t10k-images-idx3-ubyte.gz";

    image3D X_test = read_mnist_images(test_images_file);

    if (num_test_images_to_select < X_test.size()) {
        X_test.resize(num_test_images_to_select);
    }

    ConvolutionLayer conv1(output_channels, kernel_size, alpha, kernels);
    MaxPoolingLayer max_pooling(m_decimation);
    image4D batch_test = reshape_to_batch(X_test);

    image3D conv_output = conv1.forward_prop(batch_test[0]);
    int output_h = conv_output.size() / m_decimation;
    int output_w = conv_output[0].size() / m_decimation;

    image1D flattened_expected_maxpool_output = read_csv_image1D_conv("conv_output.csv");
    image3D expected_maxpool_output = reshape_to_3d_conv(flattened_expected_maxpool_output, output_h, output_w, output_channels);

    // Calculate the expected size for conv_output based on your input data and layer configuration
    int expected_maxpool_output_h = expected_maxpool_output.size();
    int expected_maxpool_output_w = expected_maxpool_output[0].size();
    int expected_maxpool_output_c = expected_maxpool_output[0][0].size();

    image3D maxpool_output = max_pooling.forward_prop(conv_output);

    // Check the size of conv_output using Google Test assertions
    ASSERT_EQ(maxpool_output.size(), expected_maxpool_output_h);
    ASSERT_EQ(maxpool_output[0].size(), expected_maxpool_output_w);
    ASSERT_EQ(maxpool_output[0][0].size(), expected_maxpool_output_c);
}

TEST(ValuesConvOutput, TestMaxPoolOutputValues) {

    constexpr int num_test_images_to_select = 1;

    std::string test_images_file = "t10k-images-idx3-ubyte.gz";

    image3D X_test = read_mnist_images(test_images_file);

    if (num_test_images_to_select < X_test.size()) {
        X_test.resize(num_test_images_to_select);
    }

    ConvolutionLayer conv1(output_channels, kernel_size, alpha, kernels);
    MaxPoolingLayer max_pooling(m_decimation);
    image4D batch_test = reshape_to_batch(X_test);

    image3D conv_output = conv1.forward_prop(batch_test[0]);
    int output_h = conv_output.size() / m_decimation;
    int output_w = conv_output[0].size() / m_decimation;

    image1D flattened_expected_maxpool_output = read_csv_image1D_conv("max_pool_output.csv");
    image3D expected_maxpool_output = reshape_to_3d_conv(flattened_expected_maxpool_output, output_h, output_w, output_channels);

    image3D maxpool_output = max_pooling.forward_prop(conv_output);

    // Check if conv_output is approximately equal to expected_conv_output element-wise within a tolerance
    double tolerance = 1e-4; // Adjust the tolerance as needed
    for (size_t h = 0; h < maxpool_output.size(); ++h) {
        for (size_t w = 0; w < maxpool_output[0].size(); ++w) {
            for (size_t c = 0; c < maxpool_output[0][0].size(); ++c) {
                ASSERT_NEAR(maxpool_output[h][w][c], expected_maxpool_output[h][w][c], tolerance);
            }
        }
    }


}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}