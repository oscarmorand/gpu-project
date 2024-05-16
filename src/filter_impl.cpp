#include "filter_impl.h"

#include <chrono>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <math.h>
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

    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        Lab* bg_mask = (Lab*) malloc(sizeof(Lab) * height * width);

        // Conversion
        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*) (buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                Lab lab = rgb_to_lab(lineptr[x]);

                bg_mask[y * width + x] = lab;
            }
        }

        if (bg_model == nullptr)
        {
            bg_model = (Lab*) malloc(sizeof(Lab) * height * width);
            std::memcpy(bg_model, bg_mask, sizeof(Lab) * height * width);
            
            for (int y = 0; y < height; ++y)
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
        else
        {
            // Update the background model
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

            float* residual_img = (float*) malloc(sizeof(float) * height * width);

            // Compute the residual image
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    Lab bg_mask_pixel = bg_mask[y * width + x];
                    Lab bg_model_pixel = bg_model[y * width + x];

                    residual_img[y * width + x] = sqrt(pow(bg_mask_pixel.L - bg_model_pixel.L, 2.0) + pow(bg_mask_pixel.a - bg_model_pixel.a, 2.0) + pow(bg_mask_pixel.b - bg_model_pixel.b, 2.0));
                }
            }

            // Save the residual image
            float max = 0.0;
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    float residual_value = residual_img[y * width + x];
                    if (residual_value > max)
                    {
                        max = residual_value;
                    }
                }
            }
            for (int y = 0; y < height; ++y)
            {
                rgb* lineptr = (rgb*) (buffer + y * stride);
                for (int x = 0; x < width; ++x)
                {
                    float residual_value = residual_img[y * width + x];
                    uint8_t out_value = static_cast<uint8_t>((residual_value / max) * 255.0);
                    lineptr[x].r = uint8_t(out_value);
                    lineptr[x].g = uint8_t(out_value);
                    lineptr[x].b = uint8_t(out_value);
                }
            }

            free(residual_img);
        }

        free(bg_mask);
    } 
}
