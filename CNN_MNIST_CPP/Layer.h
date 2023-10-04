#ifndef LAYER_H
#define LAYER_H

#include "Image.h"
#include "Patch.h"
#include "BackPropResults.h"

class Layer3D_conv
{

public:
    virtual image3D forward_prop(const image3D& image) = 0;

    virtual ConvBackPropResults back_prop(const image3D& image, const image3D& dE_dY) = 0;
    
    virtual image3D back_prop_test(const image3D& image, const image3D& dE_dY) = 0;


};

class Layer3D_maxpool
{

public:
    virtual image3D forward_prop(const image3D& image) = 0;

    virtual image3D back_prop(const image3D& image, const image3D& dE_dY) = 0;
    
    virtual image3D back_prop_test(const image3D& image, const image3D& dE_dY) = 0;


};

class Layer1D
{
public:
    virtual image1D forward_prop(const image1D& image) = 0;

    virtual FCBackPropResults back_prop(const image1D& dE_dY, const image1D& flattened_input) = 0;

    virtual image1D back_prop_test(const image1D& image, const image1D& flattened_input) = 0;

};

#endif // LAYER_H