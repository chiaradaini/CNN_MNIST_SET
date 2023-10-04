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

__global__ void convolution(const double* input, const double* kernels, int kernel_size, int input_h, int input_w, int output_h, int output_w, int input_channels, int output_channels, double* conv_output) {
	    //printf("block Idx = %d, block Dim = %d, thread Idx = %d\n", blockIdx.x, blockDim.x, threadIdx.x);
	    //printf("block Idx = %d, block Dim = %d, thread Idx = %d\n", blockIdx.y, blockDim.y, threadIdx.y);
            //printf("block Idx = %d, block Dim = %d, thread Idx = %d\n", blockIdx.z, blockDim.z, threadIdx.z);
	    int c = blockIdx.z * blockDim.z + threadIdx.z;
	    int h = blockIdx.y * blockDim.y + threadIdx.y;
	    int w = blockIdx.x * blockDim.x + threadIdx.x;
	    int output_idx = (c * output_h + h) * output_w + w;

//	    clock_t start_add = clock64();
//	    clock_t end_add = clock64();
	    if (h < output_h && w < output_w) {
		double sum = 0.0;
		for (int i = 0; i < kernel_size; ++i) {
			for (int j = 0; j < kernel_size; ++j) {
				for (int k = 0; k < input_channels; ++k) {
                			double input_pixel = input[(h + i) * (input_w * input_channels) + (w + j) * input_channels + k];
			                double kernel_value = kernels[c * (kernel_size * kernel_size * input_channels) + i * (kernel_size * input_channels) + j * input_channels + k];
					sum += input_pixel * kernel_value;
					//printf("c = %d, h = %d, w = %d, output_idx = %d, input = %f, kernel = %f, sum = %f\n", c, h, w, output_idx, input_pixel, kernel_value, sum);
					}
				}
			}
//		unsigned long micro_add = (end_add - start_add) * 1000000 / CLOCKS_PER_SEC;
//		printf("start = %llu, end = %llu, elapsed time = %llu [micro s]\n", start_add, end_add, micro_add);
		conv_output[output_idx] = sum;
		}
}

image3D conv_forward_prop(const image3D& input, const image3D& kernels, const int& numBlocks) {
    // Allocate memory on the GPU
    double *dev_conv_output;
    int input_h = input.size();
    int input_w = input[0].size();
    int input_channels = input[0][0].size();
    int kernel_size = kernels[0].size();
    int output_channels = kernels.size();
    int output_h = input_h - kernel_size + 1;
    int output_w = input_w - kernel_size + 1;
    //size_t input_size = input_h * input_w * input_channels * sizeof(double);
    //size_t kernels_size = output_channels * kernel_size * kernel_size * input_channels * sizeof(double);
    size_t conv_output_size = output_h * output_w * output_channels * sizeof(double);
    
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    size_t usedMem = totalMem - freeMem;
    //printf("Used GPU memory before allocation: %zu bytes\n", usedMem);

    //cudaMalloc((void**)&dev_input, input_size);
    //cudaMalloc((void**)&dev_kernels, kernels_size);
    cudaMalloc((void**)&dev_conv_output, conv_output_size);
    
    cudaMemGetInfo(&freeMem, &totalMem);
    usedMem = totalMem - freeMem;
    //printf("Used GPU memory after allocation: %zu bytes\n", usedMem);

    image1D flattened_input = convert_to_flattened_input(input);
    image1D flattened_kernels = convert_to_flattened_input(kernels);

    double *dev_flattened_input, *dev_flattened_kernels;
    cudaMalloc((void**)&dev_flattened_input, flattened_input.size() * sizeof(double));
    cudaMalloc((void**)&dev_flattened_kernels, flattened_kernels.size() * sizeof(double));

    cudaMemcpy(dev_flattened_input, flattened_input.data(), flattened_input.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_flattened_kernels, flattened_kernels.data(), flattened_kernels.size() * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(numBlocks, numBlocks);
    dim3 gridDim((output_w + blockDim.x - 1) / blockDim.x, (output_h + blockDim.y - 1) / blockDim.y, output_channels);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    convolution<<<blockDim, gridDim>>>(dev_flattened_input, dev_flattened_kernels, kernel_size,
                                                input_h, input_w, output_h, output_w, input_channels, output_channels, dev_conv_output);


    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.2f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    image1D conv_output_host(output_h * output_w * output_channels);
    cudaMemcpy(conv_output_host.data(), dev_conv_output, conv_output_size, cudaMemcpyDeviceToHost);

    cudaFree(dev_flattened_input);
    cudaFree(dev_flattened_kernels);
    cudaFree(dev_conv_output);

    cudaMemGetInfo(&freeMem, &totalMem);
    usedMem = totalMem - freeMem;
    //printf("Used GPU memory after deallocation: %zu bytes\n", usedMem);
    
    image3D conv_output(output_h, image2D(output_w, image1D(output_channels)));
    conv_output = convertTo3D(conv_output_host, output_h, output_w, output_channels);

    //print_kernels(conv_output);
    return conv_output;

}

