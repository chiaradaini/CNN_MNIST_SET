#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <iomanip>
#include <pthread.h>
#include <sched.h>
#include "../inc/Results.h"
#include "../inc/memory_allocation.h"
#include "../inc/Convolution.h"
#include "../inc/MaxPoolingLayer.h"
#include "../inc/FCLayer.h"
#include "../inc/Image.h"
#include "../inc/Functions.h"

// Define the CNN architecture
constexpr int kernel_size = 3;
constexpr int output_channels = 16; // Number of output channels for convolution layer
constexpr int max_pooling_decimation = 2; // Decimation rate for max pooling
constexpr int fc_input_units = 16 * 13 * 13; // Input units for the fully connected layer
constexpr int fc_output_units = 10; // Output units for the fully connected layer
constexpr double alpha = 0.01; // Learning rate
constexpr int warp = 32;
constexpr int max_num_warp = 1; // This comes from the max number of threads and blocks executable in parallel on the gpu considering the total number of pixel (10816 / 16) / 32
int granularity = 4;

int main() {

    // Set the scheduling policy to SCHED_FIFO (or SCHED_RR for round-robin)
    int policy = SCHED_FIFO;
    struct sched_param param;
    param.sched_priority = 99;  // Set the priority (0-99, 99 being the highest)

    // Set the scheduling policy and priority for the current thread (main thread)
    if (pthread_setschedparam(pthread_self(), policy, &param) != 0) {
        perror("pthread_setschedparam");
        // Handle the error if setting priority fails
        return 1;
    }

    constexpr int num_test_images_to_select = 1;

    // Specify the file paths of the MNIST dataset
    std::string test_images_file = "../data/t10k-images-idx3-ubyte.gz";
    std::string test_labels_file = "../data/t10k-labels-idx1-ubyte.gz";

    // Read the MNIST images
    image3D X_test = read_mnist_images(test_images_file);
    image1D Y_test = read_mnist_labels(test_labels_file);

    // Select the specified number of images from the test set
    if (num_test_images_to_select < X_test.size()) {
        X_test.resize(num_test_images_to_select);
        Y_test.resize(num_test_images_to_select);
    }

    // Load pre-trained weights, biases, and kernels
    image2D flattened_kernels_1 = read_csv("../data/kernels_training.csv");
    image2D weights = read_csv("../data/weights_training.csv");
    image1D biases = read_csv_image1D("../data/biases_training.csv");
    image3D kernels = reshape_to_3d(flattened_kernels_1, output_channels, kernel_size);
    image1D flattened_kernels = convert_to_flattened_input(kernels);

    // Copy kernels to the GPU
    double* dev_flattened_kernels = nullptr;
    // cudaMalloc((void**)&dev_flattened_kernels, flattened_kernels.size() * sizeof(double));
    // cudaMemcpy(dev_flattened_kernels, flattened_kernels.data(), flattened_kernels.size() * sizeof(double), cudaMemcpyHostToDevice);
    AllocateAndCopyMemory(dev_flattened_kernels, flattened_kernels.data(), flattened_kernels.size());

    // Define the Layers
    MaxPoolingLayer max_pooling(max_pooling_decimation);
    FCLayer fc_layer(fc_input_units, fc_output_units, alpha, weights, biases);

    image4D batch_test = reshape_to_batch(X_test);

    std::vector<float> executionTimes;

    // Evaluate the CNN on the test set
    int correct_predictions = 0;
    for (size_t i = 0; i < batch_test.size(); ++i) {

        int input_h = batch_test[i].size();
        int input_w = batch_test[i][0].size();
        int output_h = input_h - kernel_size + 1;
        int output_w = input_w - kernel_size + 1;
        int image_pixels = output_h * output_w * output_channels;

		for (int num_warps = 1; num_warps < (max_num_warp + 1); ++num_warps) {
			int num_blocks = static_cast<int>(ceil(static_cast<double>(image_pixels) / (warp * num_warps)));
				printf("image_pixels = %d, warp = %d, num_warps = %d, num_blocks = %d\n", image_pixels, warp, num_warps, num_blocks);
                ConvolutionResult result = conv_forward_prop(batch_test[i], dev_flattened_kernels, kernel_size, output_channels, num_blocks, warp, granularity);
				image3D conv_output = result.conv_output;
				float milliseconds = result.milliseconds;
				executionTimes.push_back(milliseconds);
				image3D max_pool_output = max_pooling.forward_prop(conv_output);
	        	image1D flatten_output = convert_to_flattened_input(max_pool_output);
	        	image1D fc_output = fc_layer.forward_prop(flatten_output);

		
        // Find the predicted label
        int predicted_label = static_cast<int>(std::distance(fc_output.begin(), std::max_element(fc_output.begin(), fc_output.end())));
        if (predicted_label == Y_test[i]) {
           correct_predictions++;

        }
        }
    }

    FreeMemory(dev_flattened_kernels);

    std::string csvFilePath = "../data/execution_times.csv";
    saveExecutionTimesToCSV(executionTimes, csvFilePath);

    double accuracy = static_cast<double>(correct_predictions) / X_test.size() * 100.0;
    std::cout << "Test accuracy: " << accuracy << "%" << std::endl;

    return 0;
}