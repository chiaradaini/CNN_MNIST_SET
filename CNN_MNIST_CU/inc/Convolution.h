// Convolution.h

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Image.h"
#include "Results.h"

// Declare the convolution CUDA kernel function
__global__ void convolution(const double* input,
                            const double* kernels,
                            int kernel_size,
                            int input_h,
                            int input_w,
                            int output_h,
                            int output_w,
                            int input_channels,
                            int output_channels,
                            int granularity,
                            double* conv_output);

// Declare the convolution forward propagation function
ConvolutionResult conv_forward_prop(const image3D& input,
                                    const double* dev_kernels,
                                    const int& kernel_size,
                                    const int& output_channels,
                                    const int& numBlocks,
                                    const int& granularity,
                                    const int& numThreads);

#endif  // CONVOLUTION_H