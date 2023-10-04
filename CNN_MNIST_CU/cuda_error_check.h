#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                            \
do {                                                                \
    cudaError_t error = call;                                       \
    if (error != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error in %s:%d - %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(error));     \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)
