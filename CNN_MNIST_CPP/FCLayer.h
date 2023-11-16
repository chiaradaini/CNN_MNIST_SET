#ifndef FCLAYER_H
#define FCLAYER_H

#include <vector>
#include <limits>
#include <cmath>
#include <random>
#include "Functions.h"
#include "Layer.h"
#include "BackPropResults.h"

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

    FCBackPropResults back_prop(const image1D& dE_dY, const image1D& flattened_input) override {
        image1D dE_dZ(output_units, 0.0);
        image2D updated_weights = weights; // Create a copy of weights
        image1D updated_biases = biases;   // Create a copy of biases
        image1D dE_dX(input_units, 0.0);
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

            // Compute gradients of output Z with respect to weights, bias, input
            image1D dZ_dw(input_units, 0.0);
            dZ_dw = flattened_input;
            double dZ_db = 1.0;
            image2D dZ_dX(weights);

            // Gradient of loss with respect to output
            image1D dE_dZ(output_units, 0.0);
            for (int j = 0; j < output_units; ++j) {
                dE_dZ[j] = dE_dY[i] * dY_dZ[j];
            }

            image2D dE_dw(input_units, image1D(output_units, 0.0));
            image1D dE_db(output_units, 0.0);
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
                    updated_weights[l][j] -= alpha * dE_dw[l][j];
                }
            }

            for (int j = 0; j < output_units; ++j) {
                updated_biases[j] -= alpha * dE_db[j];
            }

        }

        weights = updated_weights;
        biases = updated_biases;

        // Construct the struct instance with the updated values
        FCBackPropResults results;
        results.updated_weights = updated_weights;
        results.updated_biases = updated_biases;
        results.dE_dX = dE_dX;

        return results;;
    }
    
    image1D back_prop_test(const image1D& dE_dY, const image1D& flattened_input) override {
        image1D dE_dZ(output_units, 0.0);
        image1D dE_dX(input_units, 0.0);
        for (int i = 0; i < output_units; ++i) {
            if (dE_dY[i] == 0.0) {
                continue;
            }
            image1D transformation_eq(output_units, 0.0);
            for (int j = 0; j < output_units; ++j) {
                transformation_eq[j] = exp(output[j]);
                // if (transformation_eq[j] > 500) {
                //     std::cout << transformation_eq[j]  << std::endl;
                // }
            }
            // std::cout << "FC back prop - output:" << std::endl;
            // print_FCoutput(output);
            // std::cout << "FC back prop - transformation_eq:" << std::endl;
            // print_FCoutput(transformation_eq);

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
            // std::cout << "FC back prop - dY_dZ:" << std::endl;
            // print_FCoutput(dY_dZ);

            // Compute gradients of output Z with respect to weights, bias, input
            image1D dZ_dw(input_units, 0.0);
            dZ_dw = flattened_input;
            double dZ_db = 1.0;
            image2D dZ_dX(weights);

            // Gradient of loss with respect to output
            image1D dE_dZ(output_units, 0.0);
            for (int j = 0; j < output_units; ++j) {
                dE_dZ[j] = dE_dY[i] * dY_dZ[j];
            }
            // std::cout << "FC back prop - dE_dZ:" << std::endl;
            // print_FCoutput(dE_dZ);

            image2D dE_dw(input_units, image1D(output_units, 0.0));
            image1D dE_db(output_units, 0.0);
            for (int l = 0; l < input_units; ++l) {
                for (int j = 0; j < output_units; ++j) {
                    dE_dw[l][j] = dZ_dw[l] * dE_dZ[j];
                }
            }
            // std::cout << "FC back prop - dE_dw:" << std::endl;
            // print_image2D(dE_dw);

            for (int j = 0; j < output_units; ++j) {
                dE_db[j] = dE_dZ[j] * dZ_db;
                for (int l = 0; l < input_units; ++l) {
                    dE_dX[l] += dZ_dX[l][j] * dE_dZ[j];
                }
            }

            // std::cout << "FC back prop - dE_dX:" << std::endl;
            // print_FCoutput(dE_dX);

            // Update parameters
            for (int l = 0; l < input_units; ++l) {
                for (int j = 0; j < output_units; ++j){
                    weights[l][j] -= alpha * dE_dw[l][j];
                    // if (std::isnan(weights[l][j])) {
                    //     std::cout << weights[l][j] << std::endl;
                    // }
                }
            }
            // std::cout << "FC back prop - weights:" << std::endl;
            // print_image2D(weights);
            // // print_image2D(weights);

            for (int j = 0; j < output_units; ++j) {
                biases[j] -= alpha * dE_db[j];
            }
            // std::cout << "FC back prop - biases:" << std::endl;
            // print_FCoutput(biases);

        }
        return dE_dX;
    }


};

#endif // FCLAYER_H