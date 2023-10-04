#ifndef LAYER_H
#define LAYER_H

#include "Image.h"
#include "Patch.h"

class Layer3D
{

public:
    virtual image3D forward_prop(const image3D& image) = 0;
    
};

class Layer1D
{
public:
    virtual image1D forward_prop(const image1D& image) = 0;

};

#endif // LAYER_H