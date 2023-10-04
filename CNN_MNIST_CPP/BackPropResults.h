#ifndef BACKPROPRESULTS_H
#define BACKPROPRESULTS_H

#include <vector>
#include "Image.h"

struct FCBackPropResults {
    image1D dE_dX;
    image2D updated_weights;
    image1D updated_biases;
};

struct ConvBackPropResults {
    image3D dE_dX;
    image3D updated_kernels;
};

#endif // BACKPROPRESULTS_H
