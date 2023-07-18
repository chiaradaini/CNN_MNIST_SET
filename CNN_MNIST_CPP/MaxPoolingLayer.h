#ifndef MAXPOOLINGLAYER_H
#define MAXPOOLINGLAYER_H

#include <vector>
#include <limits>

#include "Layer.h"

class MaxPoolingLayer : Layer3 {
private:
    int m_decimation;
public:
    // decimation is the decimation rate of the 2D image
    MaxPoolingLayer(int decimation) : m_decimation(decimation) {}

    image3D forward_prop(const image3D& image) override {
        int image_h = image.size();
        int image_w = image[0].size();
        int channels = image[0][0].size();
        int output_h = image_h / m_decimation;
        int output_w = image_w / m_decimation;
        image3D max_pooling_output(output_h, image2D(output_w, image1D(channels, 0)));
        
        for (const auto& patch : patches_generator(m_decimation, image)) {
            const auto& patch_image = patch.patch;
            int h = patch.x;
            int w = patch.y;
            int c = patch.z;
            double max_val = patch_image[0][0][c];
            for (int i = 0; i < m_decimation; ++i) {
                for (int j = 0; j < m_decimation; ++j) {
                    max_val = std::max(max_val, patch_image[i][j][c]);
                }
            }
            max_pooling_output[h][w][c] = max_val;
        }

        return max_pooling_output;
    }

   image3D back_prop(const image3D&image, const image3D& dE_dY) {
        int image_h = image.size();
        int image_w = image[0].size();
        int channels = image[0][0].size();
        image3D dE_dk(image_h, image2D(image_w, image1D(channels, 0)));

        for (const auto& patch : patches_generator(m_decimation, image)) {
            const auto& patch_image = patch.patch;
            int h = patch.x;
            int w = patch.y;
            int c = patch.z;
            int patch_h = patch_image.size();
            int patch_w = patch_image[0].size();
            int patch_c = patch_image[0][0].size();
            image1D max_val(patch_c, std::numeric_limits<double>::min());
            for (int i = 0; i < image_h; ++i) {
                for (int j = 0; j < image_w; ++j) {
                    for (int k = 0; k < patch_c; ++k) {
                        if (patch_image[i][j][k] == max_val[k]) {
                            dE_dk[h * m_decimation + i][w * m_decimation + j][k] = dE_dY[h][w][k];
                        }
                    }
                }
            }
        }

        return dE_dk;
    }
};

#endif // MAXPOOLINGLAYER_H
