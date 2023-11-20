#include <vector>
#include <limits>
#include <vector>
#include <limits>
#include <cstdlib>
#include <tuple>
#include <random>
#include "Layer.h"
#include "Functions.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "cuda_error_check.h"
#include "Results.h"

__global__ void convolution(const double* input,
                            const double* kernels,
                            int kernel_size,
                            int input_h,
                            int input_w,
                            int output_h,
                            int output_w,
                            int input_channels,
                            int output_channels,
                            double* conv_output) {

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = output_h * output_w * output_channels;
    
    // clock_t start_add = clock64();
    // clock_t end_add = clock64();
    if (c < total_output_elements) {
        // Calculate the corresponding (h, w, c_channel) for the output
        int c_channel = c / (output_h * output_w);
        int remainder = c % (output_h * output_w);
        int w = remainder % (output_w);
        int h = remainder / (output_w);
        printf("c = %d, c_channel = %d, remainder = %d, w = %d, h = %d\n", c, c_channel, remainder, w, h);

        double sum = 0.0;

        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                for (int k = 0; k < input_channels; ++k) {
                    // Calculate input and kernel indices correctly
                    int input_idx = (h + i) * input_w * input_channels + (w + j) * input_channels + k;
                    int kernel_idx = c_channel * (kernel_size * kernel_size * input_channels) + i * (kernel_size * input_channels) + j * input_channels + k;

                    double input_pixel = input[input_idx];
                    double kernel_value = kernels[kernel_idx];
                    sum += input_pixel * kernel_value;
                    printf("c = %d, i = %d, j = %d, k = %d, input_idx = %d, input_pixel = %f, kernel_idx = %d, kernel_value = %f, sum = %f\n", c, i, j, k, input_idx, input_pixel, kernel_idx, kernel_value, sum);
                }
            }
        }
        //printf("c = %d, sum = %f\n", c, sum);
        // unsigned long micro_add = (end_add - start_add) * 1000000 / CLOCKS_PER_SEC;
		// printf("start = %llu, end = %llu, elapsed time = %llu [micro s]\n", start_add, end_add, micro_add);
        conv_output[c] = sum;
    }
}

ConvolutionResult conv_forward_prop(const image3D& input, const double* dev_kernels, const int& kernel_size, const int& output_channels, const int& numBlocks, const int& numThreads) {
    ConvolutionResult result;
    double *dev_conv_output;
    int input_h = input.size();
    int input_w = input[0].size();
    int input_channels = input[0][0].size();
    int output_h = input_h - kernel_size + 1;
    int output_w = input_w - kernel_size + 1;
    size_t conv_output_size = output_h * output_w * output_channels * sizeof(double);
 
    cudaMalloc((void**)&dev_conv_output, conv_output_size);

    image1D flattened_input = convert_to_flattened_input(input);

    double *dev_flattened_input;
    cudaMalloc((void**)&dev_flattened_input, flattened_input.size() * sizeof(double));

    cudaMemcpy(dev_flattened_input, flattened_input.data(), flattened_input.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    convolution<<<numBlocks, numThreads>>>(dev_flattened_input, dev_kernels, kernel_size,
                                                input_h, input_w, output_h, output_w, input_channels, output_channels, dev_conv_output);


    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    //printf("Kernel execution time: %.2f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    image1D conv_output_host(output_h * output_w * output_channels);
    cudaMemcpy(conv_output_host.data(), dev_conv_output, conv_output_size, cudaMemcpyDeviceToHost);

    cudaFree(dev_flattened_input);
    cudaFree(dev_conv_output);

    result.conv_output = convertTo3D(conv_output_host, output_channels, output_h, output_w);
    result.milliseconds = milliseconds;

    //std::cout << "Convolution output:" << std::endl;
    //print_kernels(result.conv_output);
    return result;

}