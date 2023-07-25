#include <iostream>
#include <vector>
#include <cmath>

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


void print_image(const image2D& image) {
    for (size_t row = 0; row < image.size(); ++row) {
        for (size_t col = 0; col < image[0].size(); ++col) {
            double pixel = image[row][col];
            std::cout << (pixel > 0.5 ? "*" : " "); // Adjust the threshold as needed for visualization
        }
        std::cout << std::endl;
    }
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

    // Print the number of images and labels read
    std::cout << "Number of training images: " << train_images_file.size() << std::endl;
    std::cout << "Number of training labels: " << train_labels_file.size() << std::endl;
    std::cout << "Number of test images: " << test_images_file.size() << std::endl;
    std::cout << "Number of test labels: " << test_labels_file.size() << std::endl;

    // Print the first image and its label
    int image_index = 0;
    int num_rows = X_train[image_index].size();
    int num_cols = X_train[image_index][0].size();

    std::cout << "Label: " << static_cast<int>(Y_train[image_index]) << std::endl;
    print_image(X_train[image_index]);
    return 0;
}