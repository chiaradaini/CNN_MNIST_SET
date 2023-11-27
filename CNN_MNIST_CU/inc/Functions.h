// MyFunctions.h

#ifndef MY_FUNCTIONS_H
#define MY_FUNCTIONS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <algorithm>
#include "Image.h"

// Function declarations
image2D read_csv(const std::string& filename);
image1D read_csv_image1D(const std::string& filename);
image3D read_mnist_images(const std::string& filename);
image1D read_mnist_labels(const std::string& filename);
void print_FCoutput(const image1D& image);
void print_kernels(const image3D& vec);
image1D convert_to_flattened_input(const image3D& image);
image3D convert_to_dE_dX_reshaped(const image1D& dE_dX, const image3D& original_shape);
image4D reshape_to_batch(const image3D& X_train);
void print_image3D(const image3D& image);
image3D reshape_to_3d(const image2D& flattened_array, int output_channels, int kernel_size);
void print_image2D(const image2D& vec);
double relu(double x);
image3D apply_relu3D(const image3D& input);
double cross_entropy_loss(const image1D& predicted_probs, const image1D& ground_truth_labels);
double calculate_cross_entropy_loss_test(const image1D& predicted_probs, const image1D& ground_truth_labels);
double calculate_accuracy(const image1D& predicted_labels, const image1D& ground_truth_labels);
image3D convertTo3D(const std::vector<double>& input, int output_channels, int output_h, int output_w);
void saveExecutionTimesToCSV(const std::vector<float>& executionTimes, const std::string& csvFilePath);

#endif // MY_FUNCTIONS_H
