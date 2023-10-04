#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <vector>
#include <cmath>
#include "Image.h"
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <algorithm>

// Function to read CSV data into an image2D
image2D read_csv(const std::string& filename) {
    image2D data;
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        image1D row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

image1D read_csv_image1D(const std::string& filename) {
    image1D data;
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            data.push_back(std::stod(cell));
        }
    }

    file.close();
    return data;
}


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

void print_FCoutput(const image1D& image) {
    for (const double& val : image) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

// Function to print the 3D vector
void print_kernels(const image3D& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t j = 0; j < vec[i].size(); ++j) {
            for (size_t k = 0; k < vec[i][j].size(); ++k) {
                std::cout << vec[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// FLatten forward function, which flatten the image form 3D to 1D, to feed the FCLayer forward
image1D convert_to_flattened_input(const image3D& image) {
    //print_kernels(image);
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
    
    //print_FCoutput(flattened_input);
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

image4D reshape_to_batch(const image3D& X_train) {
    int num_images = X_train.size();
    int image_height = X_train[0].size();
    int image_width = X_train[0][0].size();

    // Initialize the batch with zeros
    image4D batch(num_images, image3D(image_height, image2D(image_width, image1D(1, 0.0))));

    // Copy each image from X_train to the batch
    for (int i = 0; i < num_images; ++i) {
        for (int h = 0; h < image_height; ++h) {
            for (int w = 0; w < image_width; ++w) {
                batch[i][h][w][0] = X_train[i][h][w];
            }
        }
    }

    return batch;
}

// Function to print the contents of image3D with * for non-zero elements and space for zero elements
void print_image3D(const image3D& image) {
    int image_h = image.size();
    int image_w = image[0].size();
    int channels = image[0][0].size();

    for (int h = 0; h < image_h; ++h) {
        for (int i = 0; i < channels; ++i) {
            for (int w = 0; w < image_w; ++w) {
                if (image[h][w][i] != 0.0) {
                    std::cout << "*";
                } else {
                    std::cout << " "; // Two spaces to align the columns
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// Reshape a flattened 2D array back into a 3D array
image3D reshape_to_3d(const image2D& flattened_array, int output_channels, int kernel_size) {
    image3D result;

    if (flattened_array.size() != output_channels || flattened_array[0].size() != kernel_size * kernel_size) {
        // Handle error: Incorrect size of the flattened array
        return result;
    }

    result.resize(output_channels, image2D(kernel_size, image1D(kernel_size, 0.0)));

    for (int f = 0; f < output_channels; ++f) {
        int index = 0;
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                result[f][i][j] = flattened_array[f][index];
                index++;
            }
        }
    }

    return result;
}


void print_image2D(const image2D& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t j = 0; j < vec[i].size(); ++j) {
            std::cout << vec[i][j] << " ";
        }
        std::cout << std::endl;
    }
}


// ReLU activation function
double relu(double x) {
    return std::max(0.0, x);
}

// ReLU activation function for 3D images
image3D apply_relu3D(const image3D& input) {
    image3D output = input;
    for (auto& row : output) {
        for (auto& col : row) {
            for (auto& channel : col) {
                channel = relu(channel);
            }
        }
    }
    return output;
}

double cross_entropy_loss(const image1D& predicted_probs, const image1D& ground_truth_labels) {
    double loss = 0.0;
    for (size_t i = 0; i < predicted_probs.size(); ++i) {
        // Avoid taking log of 0 or 1 for numerical stability
        double epsilon = 1e-10;
        double predicted_prob = std::max(std::min(predicted_probs[i], 1.0 - epsilon), epsilon);
        loss += -ground_truth_labels[i] * std::log(predicted_prob);
    }
    return loss / predicted_probs.size(); // Normalize the loss
}

double calculate_cross_entropy_loss_test(const image1D& predicted_probs, const image1D& ground_truth_labels) {
    double loss = 0.0;
    for (size_t i = 0; i < predicted_probs.size(); ++i) {
        // Avoid taking log of 0 or 1 for numerical stability
        double epsilon = 1e-10;
        double predicted_prob = std::max(std::min(predicted_probs[i], 1.0 - epsilon), epsilon);
        loss += -ground_truth_labels[i] * std::log(predicted_prob);
    }
    return loss / predicted_probs.size(); // Normalize the loss
}

// Calculate the accuracy
double calculate_accuracy(const image1D& predicted_labels, const image1D& ground_truth_labels) {
    // Calculate the accuracy by comparing predicted_labels with ground_truth_labels
    // Count the number of correct predictions
    int correct_predictions = 0;
    for (size_t i = 0; i < predicted_labels.size(); ++i) {
        if (predicted_labels[i] == ground_truth_labels[i]) {
            correct_predictions++;
        }
    }
    return static_cast<double>(correct_predictions) / predicted_labels.size();
}


image3D convertTo3D(const std::vector<double>& input, int output_h, int output_w, int output_channels) {
    if (input.size() != output_h * output_w * output_channels) {
        // Check if the input vector size matches the desired 3D dimensions
        throw std::invalid_argument("Input vector size doesn't match the desired 3D dimensions.");
    }

    // Initialize the 3D vector with zeros
    image3D output(output_h, std::vector<std::vector<double>>(output_w, std::vector<double>(output_channels, 0.0)));

    // Fill the 3D vector with values from the input vector
    int index = 0;
    for (int c = 0; c < output_channels; ++c) {
        for (int h = 0; h < output_h; ++h) {
            for (int w = 0; w < output_w; ++w) {
                output[h][w][c] = input[index++];
            }
        }
    }

    return output;
}
#endif // FUNCTIONS_H
