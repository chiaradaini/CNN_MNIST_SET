#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H

#include <vector>
#include <limits>
#include <cstdlib>
#include <tuple>

#include "Layer.h"


class ConvolutionLayer : Layer3 {
private:
    int m_decimation;
private:
    int output_channels;
    int kernel_size;
    image3D kernels;
    image3D image;

public:
    ConvolutionLayer(int output_channels, int kernel_size) : output_channels(output_channels), kernel_size(kernel_size) {
        kernels.resize(output_channels, image2D(kernel_size, image1D(kernel_size, 0.0)));
        // Generate random filters
        for (int f = 0; f < output_channels; ++f) {
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    kernels[f][i][j] = (rand() / double(RAND_MAX)) / (kernel_size * kernel_size);
                }
            }
        }
    }

    image3D forward_prop(const image3D& image) {
        int image_h = image.size();
        int image_w = image[0].size();
        int input_channels = image[0][0].size();
        image3D convolution_output(image_h - kernel_size + 1, image2D(image_w - kernel_size + 1, image1D(output_channels, 0.0)));
        for (const auto& patch : patches_generator(m_decimation, image)) {
            const auto& patch_image = patch.patch;
            int h = patch.x;
            int w = patch.y;
            int c = patch.z;
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    convolution_output[h][w][c] += patch_image[i][j][c] * kernels[c][i][j];
                }
            }
        }

        return convolution_output;
    }

    image3D back_prop(const image3D& dE_dY, double alpha) {
        image3D dE_dk(output_channels, image2D(kernel_size, image1D(kernel_size, 0.0)));
        for (const auto& patch : patches_generator(m_decimation, image)) {
            const auto& patch_image = patch.patch;
            int h = patch.x;
            int w = patch.y;
            int c = patch.z;
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        dE_dk[c][i][j] += patch_image[i][j][c] * dE_dY[h][w][c];
                    }
                }
            }

        // Update the parameters
        for (int f = 0; f < output_channels; ++f) {
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    kernels[f][i][j] -= alpha * dE_dk[f][i][j];
                }
            }
        }

        return dE_dk;
    }
};


#endif // CONVOLUTIONLAYER_H