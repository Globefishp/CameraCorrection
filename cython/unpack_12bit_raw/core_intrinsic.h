#ifndef CORE_INTRINSIC_H
#define CORE_INTRINSIC_H

#include <stdint.h>

/**
 * @brief Unpacks a 12-bit RAW image buffer to a 16-bit buffer using SIMD instructions.
 * 
 * @param src Pointer to the source 12-bit RAW data buffer.
 * @param dst Pointer to the destination 16-bit buffer.
 * @param width The number of pixels to process. 
 *              NOTE: It is the caller's responsibility to ensure that the width is an even number.
 */
void unpack_12bit_raw(uint8_t *src, uint16_t *dst, int width);

#endif // CORE_INTRINSIC_H
