#include <iostream>
#include <vector>
#include <cmath>
#include "ConvolutionLayer.h"
#include "MaxPoolingLayer.h"
#include <fstream>
#include <vector>

std::vector<std::vector<unsigned char>> read_mnist_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open MNIST image file: " << filename << std::endl;
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

    std::vector<std::vector<unsigned char>> images(num_images, std::vector<unsigned char>(num_rows * num_cols));

    for (int i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), num_rows * num_cols);
    }

    return images;
}

std::vector<unsigned char> read_mnist_labels(const std::string& filename) {
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

    std::vector<unsigned char> labels(num_labels);

    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    return labels;
}

// Create a new function to convert unsigned char to int
std::vector<std::vector<int>> convert_to_int(const std::vector<std::vector<unsigned char>>& input) {
    std::vector<std::vector<int>> output(input.size(), std::vector<int>(input[0].size()));

    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[i].size(); ++j) {
            output[i][j] = static_cast<int>(input[i][j]);
        }
    }

    return output;
}

int main() {

    // Specify the file paths of the MNIST dataset
    std::string train_images_file = "train-images-idx3-ubyte.gz";
    std::string train_labels_file = "train-labels-idx1-ubyte.gz";
    std::string test_images_file = "t10k-images-idx3-ubyte.gz";
    std::string test_labels_file = "t10k-labels-idx1-ubyte.gz";

    // Read the MNIST images
    std::vector<std::vector<unsigned char>> X_train = read_mnist_images(train_images_file);
    std::vector<std::vector<unsigned char>> X_test = read_mnist_images(test_images_file);

    // Read the MNIST labels
    std::vector<unsigned char> Y_train = read_mnist_labels(train_labels_file);
    std::vector<unsigned char> Y_test = read_mnist_labels(test_labels_file);

    return 0;
}

