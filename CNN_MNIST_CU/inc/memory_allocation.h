#pragma once

void AllocateAndCopyMemory(double** dev_flattened_kernels, const double* flattened_kernels, size_t size);
void FreeMemory(double* dev_ptr);
