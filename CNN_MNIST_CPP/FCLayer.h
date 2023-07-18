#ifndef FCLAYER_H
#define FCLAYER_H

#include <vector>
#include <limits>
#include <cmath>

#include "Layer.h"

class FCLayer {
private:
    int m_decimation;
private:
    int input_units;
    int output_units;
    image2D weight;
    image1D bias;
    image1D flattened_input;
    image1D output;
    image3D original_shape;

public:
    FCLayer(int input_units, int output_units) : input_units(input_units), output_units(output_units) {
        weight.resize(input_units, image1D(output_units, 0.0));
        bias.resize(output_units, 0.0);
        // Initialize weights
        for (int i = 0; i < input_units; ++i) {
            for (int j = 0; j < output_units; ++j) {
                weight[i][j] = (rand() / double(RAND_MAX)) / input_units;
            }
        }
    }

    image1D forward_prop(const image3D& image) {
        int flattened_size = image.size() * image[0].size() * image[0][0].size();
        flattened_input.resize(flattened_size, 0.0);
        for (int i = 0; i < image.size(); ++i) {
            for (int j = 0; j < image[0].size(); ++j) {
                for (int k = 0; k < image[0][0].size(); ++k) {
                    flattened_input[i * image[0].size() * image[0][0].size() + j * image[0][0].size() + k] = image[i][j][k];
                }
            }
        }

        output.resize(output_units, 0.0);
        for (int j = 0; j < output_units; ++j) {
            for (int l = 0; l < input_units; ++l) {
                output[j] += flattened_input[l] * weight[l][j];
            }
            output[j] += bias[j];
        }

        // Apply softmax activation
        double max_val = std::numeric_limits<double>::lowest();
        for (double val : output) {
            max_val = std::max(max_val, val);
        }
        double exp_sum = 0.0;
        std::vector<double> softmax_output(output_units, 0.0);
        for (int i = 0; i < output_units; ++i) {
            softmax_output[i] = exp(output[i] - max_val);
            exp_sum += softmax_output[i];
        }
        for (int i = 0; i < output_units; ++i) {
            softmax_output[i] /= exp_sum;
        }

        return softmax_output;
    }

    image3D back_prop(const image1D& dE_dY, double alpha) {
        image1D dE_dZ(output_units, 0.0);
        for (int i = 0; i < output_units; ++i) {
            if (dE_dY[i] == 0.0) {
                continue;
            }
            image1D transformation_eq(output_units, 0.0);
            for (int j = 0; j < output_units; ++j) {
                transformation_eq[j] = exp(output[j]);
            }
            double S_total = 0.0;
            for (double val : transformation_eq) {
                S_total += val;
            }

            // Compute gradients with respect to output (Z)
            image1D dY_dZ(output_units, 0.0);
            for (int j = 0; j < output_units; ++j) {
                dY_dZ[j] = -transformation_eq[i] * transformation_eq[j] / (S_total * S_total);
            }
            dY_dZ[i] = transformation_eq[i] * (S_total - transformation_eq[i]) / (S_total * S_total);

            // Compute gradients of output Z with respect to weight, bias, input
            image1D dZ_dw(input_units, 0.0);
            double dZ_db = 1.0;
            image2D dZ_dX(weight);

            // Gradient of loss with respect to output
            image1D dE_dZ(output_units, 0.0);
            for (int j = 0; j < output_units; ++j) {
                dE_dZ[j] = dE_dY[i] * dY_dZ[j];
            }

            // Gradient of loss with respect to weight, bias, input
            image2D dE_dw(input_units, image1D(output_units, 0.0));
            image1D dE_db(output_units, 0.0);
            image1D dE_dX(input_units, 0.0);
            for (int l = 0; l < input_units; ++l) {
                for (int j = 0; j < output_units; ++j) {
                    dE_dw[l][j] = dZ_dw[l] * dE_dZ[j];
                }
            }
            for (int j = 0; j < output_units; ++j) {
                dE_db[j] = dE_dZ[j] * dZ_db;
                for (int l = 0; l < input_units; ++l) {
                    dE_dX[l] += dZ_dX[l][j] * dE_dZ[j];
                }
            }

            // Update parameters
            for (int l = 0; l < input_units; ++l) {
                for (int j = 0; j < output_units; ++j){
                    weight[l][j] -= alpha * dE_dw[l][j];
                }
            }
            for (int j = 0; j < output_units; ++j) {
                bias[j] -= alpha * dE_db[j];
            }
            
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
    }
};

#endif // FCLAYER_H