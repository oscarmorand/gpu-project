#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include <iostream>
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

enum morph_op
{
    EROSION,
    DILATION
};

size_t bg_model_pitch;
std::byte* bg_model = nullptr;
int n_images = 0;

template<typename T> 
__device__ T get_strided(std::byte* array, size_t pitch, int x, int y)
{
    T* lineptr = (T*) (array + y * pitch);
    return lineptr[x];
}

template<typename T> 
__device__ void set_strided(std::byte* array, size_t pitch, int x, int y, T value)
{
    T* lineptr = (T*) (array + y * pitch);
    lineptr[x] = value;
}

__device__ float f(float t)
{
    if (t > pow((6.0/29.0), 3.0))
    {
        return pow(t, (1.0/3.0));
    }
    return (1.0/3.0)*pow(29.0/6.0, 2.0)*t+(4.0/29.0);
}

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

    rgb value = get_strided<rgb>(buffer, src_stride, x, y);
    Lab lab = rgb_to_lab(value);

    set_strided<Lab>(bg_mask, bg_mask_pitch, x, y, lab);
}

__global__ void handle_first_frame(std::byte* buffer, int width, int height, int stride)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb in_val = get_strided<rgb>(buffer, stride, x, y);

    in_val.r = in_val.r / 2;
    in_val.g = in_val.g / 2;
    in_val.b = in_val.b / 2;

    set_strided<rgb>(buffer, stride, x, y, in_val);
}

__global__ void compute_residual_image(std::byte* bg_mask, std::byte* residual_img, std::byte* bg_model, int width, int height, int bg_mask_pitch, int residual_img_pitch, int bg_model_pitch)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    Lab bg_mask_pixel = get_strided<Lab>(bg_mask, bg_mask_pitch, x, y);
    Lab bg_model_pixel = get_strided<Lab>(bg_model, bg_model_pitch, x, y);

    float residual_value = sqrt(pow(bg_mask_pixel.L - bg_model_pixel.L, 2.0) + pow(bg_mask_pixel.a - bg_model_pixel.a, 2.0) + pow(bg_mask_pixel.b - bg_model_pixel.b, 2.0));
    
    set_strided<float>(residual_img, residual_img_pitch, x, y, residual_value);
}

__global__ void update_background_model(std::byte* bg_mask, std::byte* bg_model, int width, int height, int bg_mask_pitch, int bg_model_pitch, int n_frames)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    Lab bg_mask_pixel = get_strided<Lab>(bg_mask, bg_mask_pitch, x, y);
    Lab bg_model_pixel = get_strided<Lab>(bg_model, bg_model_pitch, x, y);

    bg_model_pixel.L = (bg_model_pixel.L * n_frames + bg_mask_pixel.L) / (n_frames + 1);
    bg_model_pixel.a = (bg_model_pixel.a * n_frames + bg_mask_pixel.a) / (n_frames + 1);
    bg_model_pixel.b = (bg_model_pixel.b * n_frames + bg_mask_pixel.b) / (n_frames + 1);

    set_strided<Lab>(bg_model, bg_model_pitch, x, y, bg_model_pixel);
}

__global__ void set_changed(bool* has_changed, bool val)
{
    *has_changed = val;
}

__global__ void hysteresis_threshold(std::byte* img, std::byte* out, int width, int height, size_t img_pitch, size_t out_pitch, float threshold)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    float* img_lineptr = (float*) (img + y * img_pitch);
    float in_val = img_lineptr[x];

    bool* out_lineptr = (bool*) (out + y * out_pitch);
    out_lineptr[x] = in_val > threshold;
}

#define HYST_TILE_WIDTH 34

__global__ void hysteresis_kernel(std::byte* upper, std::byte* lower, int width, int height, int upper_pitch, int lower_pitch, bool* has_changed_global)
{
    __shared__ bool upper_tile[HYST_TILE_WIDTH][HYST_TILE_WIDTH];
    __shared__ bool lower_tile[HYST_TILE_WIDTH][HYST_TILE_WIDTH];
    __shared__ bool has_changed;

    int y_pad = blockIdx.y * blockDim.y - 1;
    int x_pad = blockIdx.x * blockDim.x - 1;

    for (int tile_y = threadIdx.y; tile_y < HYST_TILE_WIDTH; tile_y += blockDim.y)
    {
        for (int tile_x = threadIdx.x; tile_x < HYST_TILE_WIDTH; tile_x += blockDim.x)
        {
            if (tile_x < width && tile_y < height && tile_x+x_pad >= 0 && tile_y+y_pad >= 0)
            {
                upper_tile[tile_y][tile_x] = get_strided<bool>(upper, upper_pitch, tile_x+x_pad, tile_y+y_pad);
                lower_tile[tile_y][tile_x] = get_strided<bool>(lower, lower_pitch, tile_x+x_pad, tile_y+y_pad);
            }
        }
    }

    has_changed = true;

    __syncthreads();

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height) return;

    while(has_changed) {
        has_changed = false;
        __syncthreads();

        if (upper_tile[y-y_pad][x-x_pad]) break;
        if (!lower_tile[y-y_pad][x-x_pad]) break;

        if (x > 0 && upper_tile[y-y_pad][x-x_pad-1]) {
            upper_tile[y-y_pad][x-x_pad] = true;
            has_changed = true;
            *has_changed_global = true;
            break;
        }

        if (x < width-1 && upper_tile[y-y_pad][x-x_pad+1]) {
            upper_tile[y-y_pad][x-x_pad] = true;
            has_changed = true;
            *has_changed_global = true;
            break;
        }

        if (y > 0) {
            if (upper_tile[y-y_pad-1][x-x_pad]) {
                upper_tile[y-y_pad][x-x_pad] = true;
                has_changed = true;
                *has_changed_global = true;
                break;
            }
        }

        if (y < height-1) {
            if (upper_tile[y-y_pad+1][x-x_pad]) {
                upper_tile[y-y_pad][x-x_pad] = true;
                has_changed = true;
                *has_changed_global = true;
                break;
            }
        }

        __syncthreads();
    }

    set_strided<bool>(upper, upper_pitch, x, y, upper_tile[y-y_pad][x-x_pad]);
}

void hysteresis(std::byte* opened_img, std::byte* hyst, int width, int height, int opened_img_pitch, int hyst_pitch)
{
    cudaError_t err;
    dim3 blockSize(32,32);
    dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);

    size_t lower_threshold_pitch;
    std::byte* lower_threshold_img;
    err = cudaMallocPitch(&lower_threshold_img, &lower_threshold_pitch, width * sizeof(bool), height);
    CHECK_CUDA_ERROR(err);

    // Lower threshold
    hysteresis_threshold<<<gridSize, blockSize>>>(opened_img, lower_threshold_img, width, height, opened_img_pitch, lower_threshold_pitch, 4.0);
    // Upper threshold
    hysteresis_threshold<<<gridSize, blockSize>>>(opened_img, hyst, width, height, opened_img_pitch, hyst_pitch, 30.0);
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);

    bool h_has_changed = true;

    bool* d_has_changed;
    err = cudaMalloc(&d_has_changed, sizeof(bool));
    CHECK_CUDA_ERROR(err);

    while (h_has_changed)
    {
        set_changed<<<1,1>>>(d_has_changed, false);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        hysteresis_kernel<<<gridSize, blockSize>>>(hyst, lower_threshold_img, width, height, hyst_pitch, lower_threshold_pitch, d_has_changed);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy(&h_has_changed, d_has_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    cudaFree(lower_threshold_img);
}

__global__ void masking_output(std::byte* src_buffer, std::byte* hyst, int width, int height, int src_stride, int hyst_pitch)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb in_val = get_strided<rgb>(src_buffer, src_stride, x, y);
    bool val = get_strided<bool>(hyst, hyst_pitch, x, y);

    in_val.r = in_val.r / 2 + (val ? 127 : 0);
    in_val.g = in_val.g / 2;
    in_val.b = in_val.b / 2;

    set_strided<rgb>(src_buffer, src_stride, x, y, in_val);
}

#define KERNEL_SIZE 5
#define HALF_KERNEL_SIZE 3
#define TILE_WIDTH 32 + 2*(HALF_KERNEL_SIZE-1)
 __device__ bool kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {0,1,1,1,0},
        {1,1,1,1,1},
        {1,1,1,1,1},
        {1,1,1,1,1},
        {0,1,1,1,0}};

__global__ void filter_morph_kernel(morph_op action, std::byte* img, std::byte* filtered_img, int width, int height, size_t img_pitch, size_t filtered_img_pitch)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    __shared__ float tile[TILE_WIDTH][TILE_WIDTH];

    int y_pad = blockIdx.y * blockDim.y - HALF_KERNEL_SIZE + 1;
    int x_pad = blockIdx.x * blockDim.x - HALF_KERNEL_SIZE + 1;

    for (int tile_y = threadIdx.y; tile_y < TILE_WIDTH; tile_y += blockDim.y)
    {
        for (int tile_x = threadIdx.x; tile_x < TILE_WIDTH; tile_x += blockDim.x)
        {
            if (tile_x+x_pad < width && tile_y+y_pad < height && tile_x+x_pad >= 0 && tile_y+y_pad >= 0)
            {
                tile[tile_y][tile_x] = get_strided<float>(img, img_pitch, tile_x+x_pad, tile_y+y_pad);
            }
        }
    }

    __syncthreads();

    float value = tile[y-y_pad][x-x_pad];

    for (int ky = -HALF_KERNEL_SIZE + 1; ky < HALF_KERNEL_SIZE; ky++){
        for (int kx = -HALF_KERNEL_SIZE + 1; kx < HALF_KERNEL_SIZE; kx++){
            if (x + kx < 0 || x + kx >= width || y + ky < 0 || y + ky >= height)
                continue;

            float k_val = tile[y + ky - y_pad][x + kx - x_pad];
            if (action == EROSION)
                value = min(value, k_val);
            else if (action == DILATION)
                value = max(value, k_val);
        }
    }

    set_strided<float>(filtered_img, filtered_img_pitch, x, y, value);
}


extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        assert(sizeof(rgb) == pixel_stride);
        cudaError_t err;
        dim3 blockSize(32,32);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);

        // Device RGB buffer
        std::byte* dBuffer;
        size_t pitch;
        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        // Device Background mask
        size_t bg_mask_pitch;
        std::byte* bg_mask;
        err = cudaMallocPitch(&bg_mask, &bg_mask_pitch, width * sizeof(Lab), height);
        CHECK_CUDA_ERROR(err);

        // Conversion from RGB to Lab color space
        convert_to_lab<<<gridSize, blockSize>>>(dBuffer, bg_mask, width, height, pitch, bg_mask_pitch);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        if (bg_model == nullptr) // First frame, no background model
        {
            // Allocation of the Background model
            err = cudaMallocPitch(&bg_model, &bg_model_pitch, width * sizeof(Lab), height);
            CHECK_CUDA_ERROR(err);
            err = cudaMemcpy2D(bg_model, bg_model_pitch, bg_mask, bg_mask_pitch, width * sizeof(Lab), height, cudaMemcpyDeviceToDevice);
            CHECK_CUDA_ERROR(err);
            
            // Fill the first frame
            handle_first_frame<<<gridSize, blockSize>>>(dBuffer, width, height, pitch);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);
        }
        else // Normal casefilter_morph_kernel
        {
            // Device residual image
            size_t residual_img_pitch;
            std::byte* residual_img;
            err = cudaMallocPitch(&residual_img, &residual_img_pitch, width * sizeof(Lab), height);
            CHECK_CUDA_ERROR(err);

            // Compute the residual image
            compute_residual_image<<<gridSize, blockSize>>>(bg_mask, residual_img, bg_model, width, height, bg_mask_pitch, residual_img_pitch, bg_model_pitch);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);

            // Update the background model with the computed background mask
            update_background_model<<<gridSize, blockSize>>>(bg_mask, bg_model, width, height, bg_mask_pitch, bg_model_pitch, n_images);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);
            n_images += 1;

            // Erosion
            size_t eroded_img_pitch;
            std::byte* eroded_img;
            err = cudaMallocPitch(&eroded_img, &eroded_img_pitch, width * sizeof(float), height);
            CHECK_CUDA_ERROR(err);
            filter_morph_kernel<<<gridSize, blockSize>>>(EROSION, residual_img, eroded_img, width, height, residual_img_pitch, eroded_img_pitch);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);
            cudaFree(residual_img);

            // Dilation
            size_t opened_img_pitch;
            std::byte* opened_img;
            err = cudaMallocPitch(&opened_img, &opened_img_pitch, width * sizeof(float), height);
            CHECK_CUDA_ERROR(err);
            filter_morph_kernel<<<gridSize, blockSize>>>(DILATION, eroded_img, opened_img, width, height, eroded_img_pitch, opened_img_pitch);            
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
            masking_output<<<gridSize, blockSize>>>(dBuffer, hyst, width, height, pitch, hyst_pitch);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);

            cudaFree(hyst);
        }

        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        cudaFree(bg_mask);
    }   
}