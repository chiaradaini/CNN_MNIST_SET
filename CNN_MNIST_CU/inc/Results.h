#ifndef RESULTS_H
#define RESULTS_H

#include <vector>
#include "Image.h"

// Define a struct to hold the results
struct ConvolutionResult {
    image3D conv_output;
    float milliseconds;
};

#endif // PRESULTS_H
