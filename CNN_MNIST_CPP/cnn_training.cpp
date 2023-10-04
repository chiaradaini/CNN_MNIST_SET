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

    // Define the number of images to select from the train and test sets
    constexpr int num_train_images_to_select = 4000;
    constexpr int num_test_images_to_select = 500;

    // Specify the file paths of the MNIST dataset
    std::string train_images_file = "train-images-idx3-ubyte.gz";
    std::string train_labels_file = "train-labels-idx1-ubyte.gz";
    std::string test_images_file = "t10k-images-idx3-ubyte.gz";
    std::string test_labels_file = "t10k-labels-idx1-ubyte.gz";

    // Read the MNIST images
    image3D X_train = read_mnist_images(train_images_file);
    image3D X_test = read_mnist_images(test_images_file);
    image1D Y_train = read_mnist_labels(train_labels_file);
    image1D Y_test = read_mnist_labels(test_labels_file);

    // Select the specified number of images from the train set
    if (num_train_images_to_select < X_train.size()) {
        X_train.resize(num_train_images_to_select);
        Y_train.resize(num_train_images_to_select);
    }

    // Select the specified number of images from the test set
    if (num_test_images_to_select < X_test.size()) {
        X_test.resize(num_test_images_to_select);
        Y_test.resize(num_test_images_to_select);
    }

    image2D flattened_kernels = read_csv("kernels.csv");
    image2D weights = read_csv("weights.csv");
    image1D biases (fc_output_units, 0.0);
    image3D kernels = reshape_to_3d(flattened_kernels, output_channels, kernel_size);

    // Define the Layers
    ConvolutionLayer conv1(output_channels, kernel_size, alpha, kernels);
    MaxPoolingLayer max_pooling(max_pooling_decimation);
    FCLayer fc_layer(fc_input_units, fc_output_units, alpha, weights, biases);

    // Train the CNN using gradient descent
    const int num_epochs = 1;

    image4D batch = reshape_to_batch(X_train);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        int total_correct_predictions = 0;
        double total_loss = 0.0;
        for (int i = 0; i < batch.size(); ++i) {

            // std::cout << "Label: " << (Y_train[i]) << std::endl;
            // print_image3D(batch[i]);

            // Forward propagation
            image3D conv_output = conv1.forward_prop(batch[i]);
            image3D max_pool_output = max_pooling.forward_prop(conv_output);
            image1D flatten_output = convert_to_flattened_input(max_pool_output);
            image1D fc_output = fc_layer.forward_prop(flatten_output);

            // Convert ground truth label to one-hot vector
            image1D ground_truth(10, 0.0);
            ground_truth[Y_train[i]] = 1.0; // Set the true label to 1.0 instead of -1.0

            // Compute the loss for the current example
            double loss = cross_entropy_loss(fc_output, ground_truth);

            // Update the total loss for the current epoch
            total_loss += loss;

            // Calculate the accuracy (1 if predicted_label matches the true label, else 0)
            int predicted_label = static_cast<int>(std::distance(fc_output.begin(), std::max_element(fc_output.begin(), fc_output.end())));
            int accuracy = (predicted_label == Y_train[i]) ? 1 : 0;

            // Update the total correct predictions for the current epoch
            total_correct_predictions += accuracy;

            // Compute the initial gradient
            image1D gradient(10, 0.0);
            gradient[Y_train[i]] = -1.0 / fc_output[Y_train[i]];

            // Backward propagation
            FCBackPropResults fc_backprop_results = fc_layer.back_prop(gradient, flatten_output);
            image1D dE_dX_fc_layer = fc_backprop_results.dE_dX;
            weights = fc_backprop_results.updated_weights;
            biases = fc_backprop_results.updated_biases;
            image3D dE_dX_reshaped = convert_to_dE_dX_reshaped(dE_dX_fc_layer, max_pool_output);
            image3D dE_dY_max_pool = max_pooling.back_prop(conv_output, dE_dX_reshaped);
            ConvBackPropResults conv_backprop_results = conv1.back_prop(batch[i], dE_dY_max_pool);
            image3D dE_dY_conv = conv_backprop_results.dE_dX;
            kernels = conv_backprop_results.updated_kernels;

            // Display accuracy and loss every 100 images
            if ((i + 1) % 100 == 0) {
                double accuracy = static_cast<double>(total_correct_predictions) / (i + 1) * 100.0;
                double average_loss = total_loss / (i+1);
                std::cout << "Epoch: " << epoch + 1 << ", Images: " << i + 1;
                std::cout << ", Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%";
                std::cout << ", Loss: " << std::fixed << std::setprecision(4) << average_loss << std::endl;
            }
                    
        }
    }

    // Flatten the 3D tensor into a 2D image (2D vector)
    image2D flattened_kernels_training;
    for (const auto& channel : kernels) {
        image1D flattened_row;
        for (const auto& row : channel) {
            flattened_row.insert(flattened_row.end(), row.begin(), row.end());
        }
        flattened_kernels_training.push_back(flattened_row);
    }

    // Save flattened_kernels to a file
    std::ofstream kernels_file("kernels_training.csv");
    if (kernels_file.is_open()) {
        for (const auto& row : flattened_kernels_training) {
            for (double value : row) {
                kernels_file << value << ",";
            }
            kernels_file << "\n";
        }
        kernels_file.close();
    } else {
        std::cerr << "Unable to open kernels file for writing." << std::endl;
    }

    // Save weights to a file
    std::ofstream weights_file("weights_training.csv");
    if (weights_file.is_open()) {
        for (const auto& row : weights) {
            for (double value : row) {
                weights_file << value << ",";
            }
            weights_file << "\n";
        }
        weights_file.close();
    } else {
        std::cerr << "Unable to open weights file for writing." << std::endl;
    }

    // Save biases to a file
    std::ofstream biases_file("biases_training.csv");
    if (biases_file.is_open()) {
        for (double value : biases) {
            biases_file << value << ",";
        }
        biases_file.close();
    } else {
        std::cerr << "Unable to open biases file for writing." << std::endl;
    }


    image4D batch_test = reshape_to_batch(X_test);

    // Evaluate the CNN on the test set
    int correct_predictions = 0;
    for (size_t i = 0; i < batch_test.size(); ++i) {
        image3D conv_output = conv1.forward_prop(batch_test[i]);
        image3D max_pool_output = max_pooling.forward_prop(conv_output);
        image1D flatten_output = convert_to_flattened_input(max_pool_output);
        image1D fc_output = fc_layer.forward_prop(flatten_output);

        // Find the predicted label
        int predicted_label = 0;
        double max_prob = fc_output[0];
        for (int j = 1; j < fc_output.size(); ++j) {
            if (fc_output[j] > max_prob) {
                max_prob = fc_output[j];
                predicted_label = j;
            }
        }

        if (predicted_label == Y_test[i]) {
            correct_predictions++;
        }
    }

    double accuracy = static_cast<double>(correct_predictions) / X_test.size() * 100.0;
    std::cout << "Test accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
