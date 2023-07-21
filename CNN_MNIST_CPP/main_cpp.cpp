#include <iostream>
#include <vector>
#include <cmath>
#include "ConvolutionLayer.h"
#include "MaxPoolingLayer.h"
#include "FCLayer.h"
#include "Image.h"
#include <fstream>
#include <vector>

// Define the CNN architecture
constexpr int input_channels = 1; // MNIST images are grayscale, so only one channel
constexpr int kernel_size = 3;
constexpr int output_channels = 32; // Number of output channels for convolution layer
constexpr int max_pooling_decimation = 2; // Decimation rate for max pooling
constexpr int fc_input_units = 32 * 13 * 13; // Input units for the fully connected layer
constexpr int fc_output_units = 10; // Output units for the fully connected layer
constexpr double alpha = 0.5;

using image1D = std::vector<double>;
using image2D = std::vector<image1D>;
using image3D = std::vector<image2D>;

image3D read_mnist_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open image file: " << filename << std::endl;
        return {};
    }

    int magic_number, num_images, num_rows, num_cols;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    image3D images(num_images, image2D(num_rows, image1D(num_cols)));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < num_rows; ++j) {
            for (int k = 0; k < num_cols; ++k) {
                unsigned char pixel;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                images[i][j][k] = static_cast<double>(pixel) / 255.0; // Normalize to [0, 1]
            }
        }
    }

    return images;
}

image1D read_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open MNIST label file: " << filename << std::endl;
        return {};
    }

    int magic_number, num_labels;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    std::vector<double> labels(num_labels);

    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = static_cast<double>(label);
    }

    return labels;
}

// FLatten forward function, which flatten the image form 3D to 1D, to feed the FCLayer forward
image1D convert_to_flattened_input(const image3D& image) {
    image1D flattened_input;
    int flattened_size = image.size() * image[0].size() * image[0][0].size();
    flattened_input.resize(flattened_size, 0.0);
    for (int i = 0; i < image.size(); ++i) {
        for (int j = 0; j < image[0].size(); ++j) {
            for (int k = 0; k < image[0][0].size(); ++k) {
                flattened_input[i * image[0].size() * image[0][0].size() + j * image[0][0].size() + k] = image[i][j][k];
            }
        }
    }
    
    return flattened_input;
}

// FLatten backword function, which rebuild the image form 1D to 3D, to feed the MaxPoolingLayer backword
image3D convert_to_dE_dX_reshaped(const image1D& dE_dX, const image3D& original_shape){
    int original_shape_h = original_shape.size();
    int original_shape_w = original_shape[0].size();
    int original_shape_channels = original_shape[0][0].size();
    image3D dE_dX_reshaped(original_shape_h, image2D(original_shape_w, image1D(original_shape_channels, 0.0)));
    size_t index = 0;
    for (size_t i = 0; i < original_shape_h; ++i) {
        for (size_t j = 0; j < original_shape_w; ++j) {
            for (size_t k = 0; k < original_shape_channels; ++k) {
                dE_dX_reshaped[i][j][k] = dE_dX[index++];
            }
        }
    }
    
    return dE_dX_reshaped;
    }

int main() {

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

    // Normalize pixel values to range [0, 1]
    //const double normalization_factor = 255.0;
    //for (auto& row : X_train) {
    //    for (int& pixel : row) {
    //        pixel /= normalization_factor;
    //    }
    //}

    //for (auto& row : X_test_int) {
    //    for (int& pixel : row) {
    //        pixel /= normalization_factor;
    //    }
    //}
    
    // Define the Layers
    ConvolutionLayer conv1(output_channels, kernel_size, alpha);
    MaxPoolingLayer max_pooling(max_pooling_decimation);
    FCLayer fc_layer(fc_input_units, fc_output_units, alpha);


    // Train the CNN using gradient descent
    const double learning_rate = 0.01;
    const int num_epochs = 10;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (size_t i = 0; i < X_train.size(); ++i) {
            // Forward propagation
            image3D conv_output = conv1.forward_prop(X_train);
            image3D max_pool_output = max_pooling.forward_prop(conv_output);
            image1D flatten_output = convert_to_flattened_input(max_pool_output);
            image1D fc_output = fc_layer.forward_prop(flatten_output);
    
            // Convert ground truth label to one-hot vector
            image1D ground_truth(10, 0.0);
            ground_truth[Y_train[i]] = 1.0;
    
            // Compute the loss and gradient of the loss function
            image1D dE_dY(fc_output.size(), 0.0);
            for (int j = 0; j < fc_output.size(); ++j) {
                dE_dY[j] = 2 * (fc_output[j] - ground_truth[j]);
            }
    
            // Backpropagation
            image1D dE_dX = fc_layer.back_prop(dE_dY);
            image3D dE_dX_reshaped = convert_to_dE_dX_reshaped(dE_dX, conv_output);
            image3D dE_dY_max_pool = max_pooling.back_prop(conv_output, dE_dX_reshaped);
            image3D dE_dY_conv = conv1.back_prop(dE_dY_max_pool);
        }
    }

    // Evaluate the CNN on the test set
    int correct_predictions = 0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        image3D conv_output = conv1.forward_prop(X_test);
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