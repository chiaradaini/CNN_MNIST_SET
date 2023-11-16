#ifndef MAXPOOLINGLAYER_H
#define MAXPOOLINGLAYER_H

#include <vector>
#include <limits>
#include "Functions.h"
#include "Layer.h"

class MaxPoolingLayer : public Layer3D_maxpool {
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
        image3D max_pooling_output(output_h, image2D(output_w, image1D(channels, 0.0)));

        for (int h = 0; h < output_h; ++h) {
            for (int w = 0; w < output_w; ++w) {
                for (int c = 0; c < channels; ++c) {
                    double max_val = image[h * m_decimation][w * m_decimation][c];
                    for (int i = 0; i < m_decimation; ++i) {
                        for (int j = 0; j < m_decimation; ++j) {
                            max_val = std::max(max_val, image[h * m_decimation + i][w * m_decimation + j][c]);
                        }
                    }
                    max_pooling_output[h][w][c] = max_val;
                }
            }
        }
        return max_pooling_output;
    }

    image3D back_prop(const image3D& image, const image3D& dE_dY) override {
        int output_h = image.size();
        int output_w = image[0].size();
        int channels = image[0][0].size();
        int dE_dY_h = dE_dY.size();
        int dE_dY_w = dE_dY[0].size();
        image3D dE_dk(output_h, image2D(output_w, image1D(channels, 0.0)));

        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < dE_dY_h; ++h) {
                for (int w = 0; w < dE_dY_w; ++w) {

                    // Create a patch
                    image2D patch(m_decimation, image1D(m_decimation, 0.0));
                    for (int i = 0; i < m_decimation; ++i) {
                        for (int j = 0; j < m_decimation; ++j) {
                            patch[i][j] = image[h * m_decimation + i][w * m_decimation + j][c];
                        }
                    }

                    // Find the maximum value in the current patch for the current channel
                    double max_val = 0.0;
                    int pos_i = 0;
                    int pos_j = 0;
                    for (int i = 0; i < m_decimation; ++i) {
                        for (int j = 0; j < m_decimation; ++j) {
                            max_val = std::max(max_val, image[h * m_decimation + i][w * m_decimation + j][c]);
                            pos_i = i;
                            pos_j = j;
                        }
                    }

                    // Assign the gradient value to the corresponding position in dE_dk
                    for (int i = 0; i < m_decimation; ++i) {
                        for (int j = 0; j < m_decimation; ++j) {
                            if (patch[i][j] == max_val) {
                                dE_dk[h * m_decimation + i][w * m_decimation + j][c] = dE_dY[h][w][c];
                            }
                        }
                    }
                }
            }
        }
        return dE_dk;
}

    image3D back_prop_test(const image3D& image, const image3D& dE_dY) override {
        int output_h = image.size();
        int output_w = image[0].size();
        int channels = image[0][0].size();
        int dE_dY_h = dE_dY.size();
        int dE_dY_w = dE_dY[0].size();
        image3D dE_dk(output_h, image2D(output_w, image1D(channels, 0.0)));

        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < dE_dY_h; ++h) {
                for (int w = 0; w < dE_dY_w; ++w) {

                    // Create a patch
                    image2D patch(m_decimation, image1D(m_decimation, 0.0));
                    for (int i = 0; i < m_decimation; ++i) {
                        for (int j = 0; j < m_decimation; ++j) {
                            patch[i][j] = image[h * m_decimation + i][w * m_decimation + j][c];
                        }
                    }

                    // Find the maximum value in the current patch for the current channel
                    double max_val = 0.0;
                    int pos_i = 0;
                    int pos_j = 0;
                    for (int i = 0; i < m_decimation; ++i) {
                        for (int j = 0; j < m_decimation; ++j) {
                            max_val = std::max(max_val, image[h * m_decimation + i][w * m_decimation + j][c]);
                            pos_i = i;
                            pos_j = j;
                        }
                    }

                    // Assign the gradient value to the corresponding position in dE_dk
                    for (int i = 0; i < m_decimation; ++i) {
                        for (int j = 0; j < m_decimation; ++j) {
                            if (patch[i][j] == max_val) {
                                dE_dk[h * m_decimation + i][w * m_decimation + j][c] = dE_dY[h][w][c];
                            }
                        }
                    }
                }
            }
        }
        return dE_dk;
}

};

#endif // MAXPOOLINGLAYER_H
