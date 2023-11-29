#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "memory_allocation.h"

void AllocateMemory(double** dev_flattened_kernels, const double* flattened_kernels, size_t size) {
    cudaMalloc((void**)dev_flattened_kernels, size * sizeof(double));

    // Check for errors
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cout << "Error allocating memory on the GPU: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    } else {
        std::cout << "Memory successfully allocated on GPU." << std::endl;
    }
}

void CopyMemoryToDevice(double** dev_flattened_kernels, const double* flattened_kernels, size_t size) {
    cudaMemcpy(*dev_flattened_kernels, flattened_kernels, size * sizeof(double), cudaMemcpyHostToDevice);

    // Check for errors
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cout << "Error copying data to GPU: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    } else {
        std::cout << "Data successfully copied to GPU." << std::endl;
    }
}

void CopyMemoryToHost(double* host_data, double** dev_data, size_t size) {
    cudaMemcpy(host_data, *dev_data, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Check for errors
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cout << "Error copying data from GPU: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    } else {
        std::cout << "Data successfully copied from GPU to host." << std::endl;
    }
}

void FreeMemory(double* dev_ptr) {
    cudaError_t cudaStatus = cudaFree(dev_ptr);

    // Check for errors
    // if (cudaStatus != cudaSuccess) {
    //     std::cout << "Error freeing GPU memory: " << cudaGetErrorString(cudaStatus) << std::endl;
    // } else {
    //     std::cout << "GPU memory successfully freed." << std::endl;
    // }
}
