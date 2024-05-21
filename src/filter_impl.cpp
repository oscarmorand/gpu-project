#include "filter_impl.h"

#include <chrono>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <iostream>
#include <algorithm>
#include "logo.h"

struct rgb {
    uint8_t r, g, b;
};

struct Lab {
    float L, a, b;
};

Lab* bg_model = nullptr;
int n_images = 0;

extern "C" {
    /*
    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*) (buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                lineptr[x].r = 0; // Back out red component

                if (x < logo_width && y < logo_height)
                {
                    float alpha = logo_data[y * logo_width + x] / 255.f;
                    lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1-alpha) * 255);
                    lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1-alpha) * 255);

                }
            }
        }

        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }   
    */

    float f(float t)
    {
        if (t > pow((6.0/29.0), 3.0))
        {
            return pow(t, (1.0/3.0));
        }
        return (1.0/3.0)*pow(29.0/6.0, 2.0)*t+(4.0/29.0);
    }

    Lab rgb_to_lab(rgb in) {
        float x = 0.4124564 * in.r + 0.3575761 * in.g + 0.1804375 * in.b;
        float y = 0.2126729 * in.r + 0.7151522 * in.g + 0.0721750 * in.b;
        float z = 0.0193339 * in.r + 0.1191920 * in.g + 0.9503041 * in.b;

        float xn = 95.0489;
        float yn = 100.0;
        float zn = 108.8840;

        float f_y_over_yn = f(y / yn);

        float L = 116 * f_y_over_yn - 16;
        float a = 500 * (f(x / xn) - f_y_over_yn);
        float b = 200 * (f_y_over_yn - f(z / zn));

        return {L, a, b};
    }

    bool* hysteresis(float* residual_img, bool* upper_threshold_vals, int width, int height)
    {
        float upper_threshold = 30;
        float lower_threshold = 4;

        bool* lower_threshold_vals = (bool*) malloc(sizeof(bool) * height * width);

        // Compute the hysteresis thresholds
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float value = residual_img[y * width + x];
                upper_threshold_vals[y * width + x] = value > upper_threshold;
                lower_threshold_vals[y * width + x] = value > lower_threshold;
            }
        }

        // Compute the hysteresis propagation
        bool has_changed = true;
        while (has_changed)
        {
            has_changed = false;
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    if (upper_threshold_vals[y * width + x])
                    {
                        if (x > 0 && !upper_threshold_vals[y * width + x - 1] && lower_threshold_vals[y * width + x - 1])
                        {
                            has_changed = true;
                            upper_threshold_vals[y * width + x - 1] = true;
                        }
                        if (x+1 < width && !upper_threshold_vals[y * width + x + 1] && lower_threshold_vals[y * width + x + 1])
                        {
                            has_changed = true;
                            upper_threshold_vals[y * width + x + 1] = true;
                        }
                        if (y > 0 && !upper_threshold_vals[(y-1) * width + x] && lower_threshold_vals[(y-1) * width + x])
                        {
                            has_changed = true;
                            upper_threshold_vals[(y-1) * width + x] = true;
                        }
                        if (y+1 < height && !upper_threshold_vals[(y+1) * width + x] && lower_threshold_vals[(y+1) * width + x])
                        {
                            has_changed = true;
                            upper_threshold_vals[(y+1) * width + x] = true;
                        }
                    }
                }
            }
        }

        free(lower_threshold_vals);

        return upper_threshold_vals;
    }

    #define KERNEL_SIZE 5
    bool kernel[KERNEL_SIZE][KERNEL_SIZE] = {
            {0,1,1,1,0},
            {1,1,1,1,1},
            {1,1,1,1,1},
            {1,1,1,1,1},
            {0,1,1,1,0}};

    enum morph_op
    {
        EROSION,
        DILATION
    };

    void filter_morph(morph_op action, float* residual_img, float* residual_img_filtered, int width, int height)
    {
        int half_size = 3;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float value = residual_img[y * width + x];

                for (int ky = -half_size; ky < half_size - 1; ++ky){
                    for (int kx = -half_size; kx < half_size - 1; ++kx) {
                        if (kernel[ky + half_size][kx + half_size]) {

                            int yy = std::min(std::max(y + ky, 0), height - 1);
                            int xx = std::min(std::max(x + kx, 0), width - 1);

                            if (action == EROSION)
                                value = fmin(value, residual_img[yy * width + xx]);
                            else if (action == DILATION)
                                value = fmax(value, residual_img[yy * width + xx]);
                        }
                    }
                }
                residual_img_filtered[y * width + x] = value;
            }
        }
    }

    void convert_to_lab(uint8_t* buffer, Lab* bg_mask, int width, int height, int stride)
    {
        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*) (buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                Lab lab = rgb_to_lab(lineptr[x]);

                bg_mask[y * width + x] = lab;
            }
        }
    }

    void handle_first_frame(uint8_t* buffer, Lab* bg_mask, int width, int height, int stride)
    {
        bg_model = (Lab*) malloc(sizeof(Lab) * height * width);
        std::memcpy(bg_model, bg_mask, sizeof(Lab) * height * width);
            
        for (int y = 0; y < height; ++y) // Set the output to black
        {
            rgb* lineptr = (rgb*) (buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                lineptr[x].r = 0;
                lineptr[x].g = 0;
                lineptr[x].b = 0;
            }
        }
    }

    void update_background_model(Lab* bg_mask, int width, int height)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                Lab bg_mask_pixel = bg_mask[y * width + x];
                Lab bg_model_pixel = bg_model[y * width + x];

                bg_model_pixel.L = (bg_model_pixel.L * n_images + bg_mask_pixel.L) / (n_images + 1);
                bg_model_pixel.a = (bg_model_pixel.a * n_images + bg_mask_pixel.a) / (n_images + 1);
                bg_model_pixel.b = (bg_model_pixel.b * n_images + bg_mask_pixel.b) / (n_images + 1);

                n_images += 1;

                bg_model[y * width + x] = bg_model_pixel;
            }
        }
    }

    void compute_residual_image(Lab* bg_mask, float* residual_img, int width, int height)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                Lab bg_mask_pixel = bg_mask[y * width + x];
                Lab bg_model_pixel = bg_model[y * width + x];

                residual_img[y * width + x] = sqrt(pow(bg_mask_pixel.L - bg_model_pixel.L, 2.0) + pow(bg_mask_pixel.a - bg_model_pixel.a, 2.0) + pow(bg_mask_pixel.b - bg_model_pixel.b, 2.0));
            }
        }
    }

    void masking_output(uint8_t* buffer, bool* hyst, int width, int height, int stride)
    {
        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*) (buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                rgb in_val = lineptr[x];
                bool val = hyst[y * width + x];

                lineptr[x].r = in_val.r / 2 + (val ? 127 : 0);
                lineptr[x].g = in_val.g / 2;
                lineptr[x].b = in_val.b / 2;
            }
        }
    }

    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        Lab* bg_mask = (Lab*) malloc(sizeof(Lab) * height * width);

        // Conversion from RGB to Lab color space
        convert_to_lab(buffer, bg_mask, width, height, stride);

        if (bg_model == nullptr) // First frame, no background model
        {
            handle_first_frame(buffer, bg_mask, width, height, stride);
        }
        else // Normal case
        {
            float* residual_img = (float*) malloc(sizeof(float) * height * width);

            // Compute the residual image
            compute_residual_image(bg_mask, residual_img, width, height);

            // Update the background model with the computed background mask
            update_background_model(bg_mask, width, height);
            
            float* eroded_img = (float*) malloc(sizeof(float) * height * width);
            filter_morph(EROSION, residual_img, eroded_img, width, height);
            free(residual_img);

            float* opened_img = (float*) malloc(sizeof(float) * height * width);
            filter_morph(DILATION, eroded_img, opened_img, width, height);
            free(eroded_img);

            bool* hyst = (bool*) malloc(sizeof(bool) * height * width);
            hysteresis(opened_img, hyst, width, height);
            free(opened_img);

            // Save the hysteresis
            masking_output(buffer, hyst, width, height, stride);

            free(hyst);
        }

        free(bg_mask);
    } 
}
