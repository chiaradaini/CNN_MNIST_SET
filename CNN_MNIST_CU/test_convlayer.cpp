#include <gtest/gtest.h>
#include "ConvolutionLayer.h"
#include "Functions.h"

// Define the CNN architecture
constexpr int kernel_size = 3;
constexpr int output_channels = 16; // Number of output channels for convolution layer
constexpr double alpha = 0.01; //learning rate

// Load pre-trained weights, biases, and kernels
image2D flattened_kernels = read_csv("kernels_training.csv");
image3D kernels = reshape_to_3d(flattened_kernels, output_channels, kernel_size);

TEST(VariableSizeTest, TestConvOutputSize) {

    constexpr int num_test_images_to_select = 1;

    std::string test_images_file = "t10k-images-idx3-ubyte.gz";

    image3D X_test = read_mnist_images(test_images_file);

    if (num_test_images_to_select < X_test.size()) {
        X_test.resize(num_test_images_to_select);
    }

    ConvolutionLayer conv1(output_channels, kernel_size, alpha, kernels);
    image4D batch_test = reshape_to_batch(X_test);

    int output_h = batch_test[0].size() - kernel_size + 1;
    int output_w = batch_test[0][0].size() - kernel_size + 1;

    image1D flattened_expected_conv_output = read_csv_image1D_conv("conv_output.csv");
    image3D expected_conv_output = reshape_to_3d_conv(flattened_expected_conv_output, output_h, output_w, output_channels);

    // Calculate the expected size for conv_output based on your input data and layer configuration
    int expected_conv_output_h = expected_conv_output.size();
    int expected_conv_output_w = expected_conv_output[0].size();
    int expected_conv_output_c = output_channels;

    // Calculate the size of conv_output
    image3D conv_output = conv1.forward_prop(batch_test[0]);

    // Check the size of conv_output using Google Test assertions
    ASSERT_EQ(conv_output.size(), expected_conv_output_h);
    ASSERT_EQ(conv_output[0].size(), expected_conv_output_w);
    ASSERT_EQ(conv_output[0][0].size(), expected_conv_output_c);
}

TEST(ValuesConvOutput, TestConvOutputValues) {

    constexpr int num_test_images_to_select = 1;

    std::string test_images_file = "t10k-images-idx3-ubyte.gz";

    image3D X_test = read_mnist_images(test_images_file);

    if (num_test_images_to_select < X_test.size()) {
        X_test.resize(num_test_images_to_select);
    }

    ConvolutionLayer conv1(output_channels, kernel_size, alpha, kernels);
    image4D batch_test = reshape_to_batch(X_test);

    int output_h = batch_test[0].size() - kernel_size + 1;
    int output_w = batch_test[0][0].size() - kernel_size + 1;

    image1D flattened_expected_conv_output = read_csv_image1D_conv("conv_output.csv");
    image3D expected_conv_output = reshape_to_3d_conv(flattened_expected_conv_output, output_h, output_w, output_channels);
    image3D conv_output = conv1.forward_prop(batch_test[0]);
    
    // Check if conv_output is approximately equal to expected_conv_output element-wise within a tolerance
    double tolerance = 1e-4; // Adjust the tolerance as needed
    for (size_t h = 0; h < conv_output.size(); ++h) {
        for (size_t w = 0; w < conv_output[0].size(); ++w) {
            for (size_t c = 0; c < conv_output[0][0].size(); ++c) {
                ASSERT_NEAR(conv_output[h][w][c], expected_conv_output[h][w][c], tolerance);
            }
        }
    }


}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}