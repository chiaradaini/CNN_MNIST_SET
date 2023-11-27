#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "memory_allocation.h"

void AllocateAndCopyMemory(double* dev_flattened_kernels, const double* flattened_kernels, size_t size) {
    
    cudaMalloc((void**)&dev_flattened_kernels, size * sizeof(double));
    cudaMemcpy(dev_flattened_kernels, flattened_kernels, size * sizeof(double), cudaMemcpyHostToDevice);

    // Check for errors
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cout << "Error copying memory to GPU: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    } else {
        std::cout << "Memory successfully copied to GPU." << std::endl;
    }
}

void FreeMemory(double* dev_ptr) {
    cudaError_t cudaStatus = cudaFree(dev_ptr);
    
    if (cudaStatus != cudaSuccess) {
        std::cout << "Error freeing GPU memory: " << cudaGetErrorString(cudaStatus) << std::endl;
    } else {
        std::cout << "GPU memory successfully freed." << std::endl;
    }
}
