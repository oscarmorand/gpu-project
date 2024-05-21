#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include "logo.h"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

struct rgb {
    uint8_t r, g, b;
};

struct Lab {
    float L, a, b;
};

int bg_model_pitch;
std::byte* bg_model = nullptr;
int n_images = 0;

__constant__ uint8_t* logo;

/// @brief Black out the red channel from the video and add EPITA's logo
/// @param buffer 
/// @param width 
/// @param height 
/// @param stride 
/// @param pixel_stride 
/// @return 
__global__ void remove_red_channel_inp(std::byte* buffer, int width, int height, int stride)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return; 

    rgb* lineptr = (rgb*) (buffer + y * stride);
    if (y < logo_height && x < logo_width) {
        float alpha = logo[y * logo_width + x] / 255.f;
        lineptr[x].r = 0;
        lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1-alpha) * 255);
        lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1-alpha) * 255);
    } else {
        lineptr[x].r = 0;
    }
}

namespace 
{
    void load_logo()
    {
        static auto buffer = std::unique_ptr<std::byte, decltype(&cudaFree)>{nullptr, &cudaFree}; 

        if (buffer == nullptr)
        {
            cudaError_t err;
            std::byte* ptr;
            err = cudaMalloc(&ptr, logo_width * logo_height);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpy(ptr, logo_data, logo_width * logo_height, cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpyToSymbol(logo, &ptr, sizeof(ptr));
            CHECK_CUDA_ERROR(err);

            buffer.reset(ptr);
        }

    }
}

/*
extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        load_logo();

        assert(sizeof(rgb) == pixel_stride);
        std::byte* dBuffer;
        size_t pitch;

        cudaError_t err;
        
        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(16,16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);

        remove_red_channel_inp<<<gridSize, blockSize>>>(dBuffer, width, height, pitch);

        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);


        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }   
}
*/

__device__ Lab rgb_to_lab(rgb in) 
{
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

__global__ void convert_to_lab(std::byte* buffer, std::byte* bg_mask, int width, int height, int src_stride, int bg_mask_pitch)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb* buffer_lineptr = (rgb*) (buffer + y * src_stride);

    Lab lab = rgb_to_lab(buffer_lineptr[x]);

    Lab* bg_mask_lineptr = (Lab*) (bg_mask + y * bg_mask_pitch);
    bg_mask_lineptr[x] = lab;
}

__global__ void handle_first_frame(std::byte* buffer, int width, int height, int stride)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb* lineptr = (rgb*) (buffer + y * stride);

    lineptr[x] = {0,0,0};
}

__global__ void compute_residual_image(std::byte* bg_mask, std::byte* residual_img, int width, int height, int bg_mask_pitch, int residual_img_pitch)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    Lab* bg_mask_lineptr = (Lab*) (bg_mask + y * bg_mask_pitch);
    Lab bg_mask_pixel = bg_mask_lineptr[x];

    Lab* bg_model_lineptr = (Lab*) (bg_model + y * bg_model_pitch);
    Lab bg_model_pixel = bg_model_lineptr[x];

    float* residual_img_lineptr = (float*) (residual_img + y * residual_img_pitch);
    residual_img_lineptr[x] = sqrt(pow(bg_mask_pixel.L - bg_model_pixel.L, 2.0) + pow(bg_mask_pixel.a - bg_model_pixel.a, 2.0) + pow(bg_mask_pixel.b - bg_model_pixel.b, 2.0));
}

__global__ void update_background_model(std::byte* bg_mask, int width, int height, int bg_mask_pitch, int n_frames)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    Lab* bg_mask_lineptr = (Lab*) (bg_mask + y * bg_mask_pitch);
    Lab bg_mask_pixel = bg_mask_lineptr[x];

    Lab* bg_model_lineptr = (Lab*) (bg_model + y * bg_model_pitch);
    Lab bg_model_pixel = bg_model_lineptr[x];

    bg_model_pixel.L = (bg_model_pixel.L * n_frames + bg_mask_pixel.L) / (n_frames + 1);
    bg_model_pixel.a = (bg_model_pixel.a * n_frames + bg_mask_pixel.a) / (n_frames + 1);
    bg_model_pixel.b = (bg_model_pixel.b * n_frames + bg_mask_pixel.b) / (n_frames + 1);

    bg_model_lineptr[x] = bg_model_pixel;
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

__global__ void filter_morph(morph_op action, std::byte* residual_img, std::byte* eroded_img, int width, int height, int residual_img_pitch, int eroded_img_pitch)
{
    // TODO Stencil pattern
}

void hysteresis(std::byte* opened_img, std::byte* hyst, int width, int height, int opened_img_pitch, int hyst_pitch)
{
    // TODO Map + Stencil pattern (call two kernels)
    dim3 blockSize(16,16);
    dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);
}

__global__ void masking_output(std::byte* src_buffer, std::byte* hyst, int width, int height, int src_stride, int hyst_pitch)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb* buffer_lineptr = (rgb*) (src_buffer + y * src_stride);
    rgb in_val = buffer_lineptr[x];

    bool* hyst_lineptr = (bool*) (hyst + y * hyst_pitch);
    bool val = hyst_lineptr[x];

    buffer_lineptr[x].r = in_val.r / 2 + (val ? 127 : 0);
    buffer_lineptr[x].g = in_val.g / 2;
    buffer_lineptr[x].b = in_val.b / 2;
}

extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        assert(sizeof(rgb) == pixel_stride);
        cudaError_t err;

        dim3 blockSize(16,16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);
        
        size_t bg_mask_pitch;
        std::byte* bg_mask;
        err = cudaMallocPitch(&bg_mask, &bg_mask_pitch, width * sizeof(Lab), height);
        CHECK_CUDA_ERROR(err);

        // Conversion from RGB to Lab color space
        convert_to_lab<<<gridSize, blockSize>>>(src_buffer, bg_mask, width, height, src_stride, bg_mask_pitch);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        if (bg_model == nullptr) // First frame, no background model
        {
            err = cudaMallocPitch(&bg_model, &bg_model_pitch, width * sizeof(Lab), height);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpy2D(bg_model, bg_model_pitch, bg_mask, bg_mask_pitch, width * sizeof(Lab), height, cudaMemcpyDeviceToDevice);
            CHECK_CUDA_ERROR(err);

            handle_first_frame<<<gridSize, blockSize>>>(src_buffer, width, height, src_stride);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);
        }
        else // Normal case
        {
            size_t residual_img_pitch;
            std::byte* residual_img;
            err = cudaMallocPitch(&residual_img, &residual_img_pitch, width * sizeof(Lab), height);
            CHECK_CUDA_ERROR(err);

            // Compute the residual image
            compute_residual_image<<<gridSize, blockSize>>>(bg_mask, residual_img, width, height, bg_mask_pitch, residual_img_pitch);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);

            // Update the background model with the computed background mask
            update_background_model<<<gridSize, blockSize>>>(bg_mask, width, height, bg_mask_pitch, n_images);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);
            n_images += 1;

            // Erosion
            size_t eroded_img_pitch;
            std::byte* eroded_img;
            err = cudaMallocPitch(&eroded_img, &eroded_img_pitch, width * sizeof(float), height);
            CHECK_CUDA_ERROR(err);
            filter_morph<<<gridSize, blockSize>>>(EROSION, residual_img, eroded_img, width, height, residual_img_pitch, eroded_img_pitch);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);
            cudaFree(residual_img);

            // Dilation
            size_t opened_img_pitch;
            std::byte* opened_img;
            err = cudaMallocPitch(&opened_img, &opened_img_pitch, width * sizeof(float), height);
            CHECK_CUDA_ERROR(err);
            filter_morph<<<gridSize, blockSize>>>(DILATION, residual_img, opened_img, width, height, residual_img_pitch, opened_img_pitch);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);
            cudaFree(eroded_img);

            // Hysteresis
            size_t hyst_pitch;
            std::byte* hyst;
            err = cudaMallocPitch(&hyst, &hyst_pitch, width * sizeof(bool), height);
            CHECK_CUDA_ERROR(err);
            hysteresis(opened_img, hyst, width, height, opened_img_pitch, hyst_pitch);
            cudaFree(opened_img);

            // Save the mask
            masking_output<<<gridSize, blockSize>>>(src_buffer, hyst, width, height, src_stride, hyst_pitch);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);

            cudaFree(hyst);
        }

        cudaFree(bg_mask);
    }   
}