#ifndef FCLAYER_H
#define FCLAYER_H

#include <vector>
#include <limits>
#include <cmath>
#include <random>
#include "Functions.h"
#include "Layer.h"

class FCLayer : public Layer1D {
private:
    int input_units;
    int output_units;
    double alpha;
    image2D weights;
    image1D biases;
    image1D output;
    image3D original_shape;

public:
    FCLayer(int input_units, int output_units, double alpha, image2D weights, image1D biases) : input_units(input_units), output_units(output_units), alpha(alpha), weights(weights), biases(biases) {
        output.resize(output_units, 0.0);
        // weights.resize(input_units, image1D(output_units, 0.0));
        // biases.resize(output_units, 0.0);

        // // Random number generator with a normal distribution
        // std::default_random_engine generator;
        // std::normal_distribution<double> distribution(0.0, 1.0);

        // // Generate random numbers and store them in the output vector
        // for (int i = 0; i < input_units; ++i) {
        //     for (int j = 0; j < output_units; ++j) {
        //         weights[i][j] = distribution(generator);
        //     }
        // }
    }

    image1D forward_prop(const image1D& flattened_input) override {
        
        image1D first_output(output_units, 0.0);
        // print_FCoutput(first_output);
        // print_FCoutput(output);
        for (int j = 0; j < output_units; ++j) {
            for (int l = 0; l < input_units; ++l) {
                first_output[j] += flattened_input[l] * weights[l][j];
            }
            first_output[j] += biases[j];
        }
        output = first_output;
        // print_FCoutput(first_output);
        // print_FCoutput(output);

        // Apply softmax activation
        double max_val = std::numeric_limits<double>::lowest();
        for (double val : output) {
            max_val = std::max(max_val, val);
        }
        double exp_sum = 0.0;
        image1D softmax_output(output_units, 0.0);
        for (int i = 0; i < output_units; ++i) {
            softmax_output[i] = exp(output[i] - max_val);
            exp_sum += softmax_output[i];
        }
        for (int i = 0; i < output_units; ++i) {
            softmax_output[i] /= exp_sum;
        }
        return softmax_output;

    }

};

#endif // FCLAYER_H