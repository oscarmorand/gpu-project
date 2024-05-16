#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void filter_impl(uint8_t* buffer, int width, int height, int plane_stride, int pixel_stride);

#ifdef __cplusplus
}
#endif
