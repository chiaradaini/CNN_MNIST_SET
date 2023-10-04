#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H

#include <vector>
#include <limits>
#include <cstdlib>
#include <tuple>
#include <random>
#include "Layer.h"
#include "Functions.h"
#include "BackPropResults.h"

class ConvolutionLayer : public Layer3D_conv {
private:
    int output_channels;
    int kernel_size;
    double alpha;
    image3D kernels;
    image3D image;

public:
    ConvolutionLayer(int output_channels, int kernel_size, double alpha, image3D kernels) : output_channels(output_channels), kernel_size(kernel_size), alpha(alpha), kernels(kernels) {
        // kernels.resize(output_channels, image2D(kernel_size, image1D(kernel_size, 0.0)));
        // // Random number generator with a normal distribution
        // std::default_random_engine generator;
        // std::normal_distribution<double> distribution(0.0, 1.0);

        // for (int f = 0; f < output_channels; ++f) {
        //     for (int i = 0; i < kernel_size; ++i) {
        //         for (int j = 0; j < kernel_size; ++j) {
        //             kernels[f][i][j] = distribution(generator);;
        //         }
        //     }
        // }
    }

    image3D forward_prop(const image3D& image) override {
    int image_h = image.size();
    int image_w = image[0].size();
    int input_channels = image[0][0].size();
    int output_h = image_h - kernel_size + 1;
    int output_w = image_w - kernel_size + 1;

    image2D convolution_output_sum(kernel_size, image1D(kernel_size, 0.0));
    image3D convolution_output(output_h, image2D(output_w, image1D(output_channels, 0.0)));

    for (int c = 0; c < output_channels; ++c){
        for (int h = 0; h < output_h; ++h) {
            for (int w = 0; w < output_w; ++w) {
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        for (int k = 0; k < input_channels; ++k) {
                            convolution_output_sum[i][j] = image[h + i][w + j][k] * kernels[c][i][j];
                        }
                    }
                }
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        convolution_output[h][w][c] += convolution_output_sum[i][j];
                    }
                }
            }
        }
    }
    return convolution_output;
}

    ConvBackPropResults back_prop(const image3D& input, const image3D& dE_dY) override {
    int input_h = input.size();
    int input_w = input[0].size();
    int input_channels = input[0][0].size();
    int output_h = dE_dY.size();
    int output_w = dE_dY[0].size();

    image3D updated_kernels = kernels;
    image3D dE_dX(input_h, image2D(input_w, image1D(input_channels, 0.0)));

    for (int c = 0; c < output_channels; ++c) {
        for (int h = 0; h < output_h; ++h) {
            for (int w = 0; w < output_w; ++w) {
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        for (int k = 0; k < input_channels; ++k) {
                            dE_dX[h + i][w + j][k] += dE_dY[h][w][c] * updated_kernels[c][i][j];
                            updated_kernels[c][i][j] -= alpha * dE_dY[h][w][c] * input[h + i][w + j][k];
                        }
                    }
                }
            }
        }
    }

    kernels = updated_kernels;

    // Construct the struct instance with the updated values
    ConvBackPropResults results;
    results.updated_kernels = updated_kernels;
    results.dE_dX = dE_dX;

    return results;
    }

    image3D back_prop_test(const image3D& input, const image3D& dE_dY) override {
        int input_h = input.size();
        int input_w = input[0].size();
        int input_channels = input[0][0].size();
        int output_h = dE_dY.size();
        int output_w = dE_dY[0].size();

        image3D dE_dX(input_h, image2D(input_w, image1D(input_channels, 0.0)));

        for (int c = 0; c < output_channels; ++c) {
            for (int h = 0; h < output_h; ++h) {
                for (int w = 0; w < output_w; ++w) {
                    for (int i = 0; i < kernel_size; ++i) {
                        for (int j = 0; j < kernel_size; ++j) {
                            for (int k = 0; k < input_channels; ++k) {
                                dE_dX[h + i][w + j][k] += dE_dY[h][w][c] * kernels[c][i][j];
                                kernels[c][i][j] -= alpha * dE_dY[h][w][c] * input[h + i][w + j][k];
                                // if (std::isnan(kernels[c][i][j])) {
                                //     std::cout << kernels[c][i][j] << std::endl;
                                // }
                            }
                        }
                    }
                }
            }
        }
    return dE_dX;
    }
};


#endif // CONVOLUTIONLAYER_H