#pragma once

void AllocateMemory(double** dev_flattened_kernels, const double* flattened_kernels, size_t size);
void CopyMemoryToDevice(double** dev_flattened_kernels, const double* flattened_kernels, size_t size);
void CopyMemoryToHost(double* host_data, double** dev_data, size_t size);
void FreeMemory(double* dev_ptr);
