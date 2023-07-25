#ifndef LAYER_H
#define LAYER_H

#include "Image.h"
#include "Patch.h"

class Layer3D
{
// protected:
//     int patch_size;
public:
    virtual image3D forward_prop(const image3D& image) = 0;

    virtual image3D back_prop(const image3D& image) = 0;
    
    static patches patches_generator(const int patch_size, const image3D& image) {
        patches patches;
        int output_h = image.size() - patch_size + 1;
        int output_w = image[0].size() - patch_size + 1;
        int n_channels = image[0][0].size();

        // Iterate over the channels in the 3D image
        for (int c = 0; c < n_channels; ++ c) {
            for (int h = 0; h < (output_h - patch_size + 1); ++h) {
                for (int w = 0; w < (output_w - patch_size + 1); ++w) {
                    image3D patch_image(patch_size, image2D(patch_size, image1D(n_channels, 0.0)));
                    for (int i = 0; i < patch_size; ++i) {
                        for (int j = 0; j < patch_size; ++j) {
                            patch_image[i][j][c] = image[h + patch_size][w + patch_size][c];
                        }
                    }
                    Patch patch = { patch_image, h, w, c };
                    patches.emplace_back(patch);
                }
            }
        }

        return patches;
    }
};

class Layer3D3D
{
// protected:
//     int patch_size;
public:
    virtual image3D forward_prop(const image3D& image) = 0;

    virtual image3D back_prop(const image3D& image, const image3D& dE_dY) = 0;
    
    static patches patches_generator(const int patch_size, const image3D& image) {
        patches patches;
        int output_h = image.size() - patch_size + 1;
        int output_w = image[0].size() - patch_size + 1;
        int n_channels = image[0][0].size();

        // Iterate over the channels in the 3D image
        for (int c = 0; c < n_channels; ++ c) {
            for (int h = 0; h < (output_h - patch_size + 1); ++h) {
                for (int w = 0; w < (output_w - patch_size + 1); ++w) {
                    image3D patch_image(patch_size, image2D(patch_size, image1D(n_channels, 0.0)));
                    for (int i = 0; i < patch_size; ++i) {
                        for (int j = 0; j < patch_size; ++j) {
                            patch_image[i][j][c] = image[h + patch_size][w + patch_size][c];
                        }
                    }
                    Patch patch = { patch_image, h, w, c };
                    patches.emplace_back(patch);
                }
            }
        }

        return patches;
    }
};

class Layer1D
{
public:
    virtual image1D forward_prop(const image1D& image) = 0;

    virtual image1D back_prop(const image1D& image) = 0;
};

#endif // LAYER_H