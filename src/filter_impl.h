#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void filter_impl(uint8_t* buffer, int width, int height, int plane_stride, int pixel_stride, int th_low, int th_high);

#ifdef __cplusplus
}
#endif
