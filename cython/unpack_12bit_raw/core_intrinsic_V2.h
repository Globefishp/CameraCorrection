#ifndef CORE_INTRINSIC_V2_H
#define CORE_INTRINSIC_V2_H

#include <stdint.h>

/**
 * @brief Unpacks a 12-bit RAW image buffer to a 16-bit buffer using SIMD instructions
 *        with a 2-stage software pipeline and streaming stores.
 * 
 * This function processes the image data in chunks of 32 pixels (48 bytes) for optimal
 * performance. The main loop is software-pipelined to overlap memory I/O with computation,
 * hiding latency and improving throughput. Any remaining pixels at the end are handled 
 * by a standard C loop.
 * 
 * @param src Pointer to the source 12-bit RAW data buffer.
 * @param dst Pointer to the destination 16-bit buffer. The destination address MUST be 
 *            16-byte aligned for streaming stores to work correctly.
 * @param width The number of pixels to process. 
 *              NOTE: It is the caller's responsibility to ensure that the width is an even number.
 */
void unpack_12bit_raw(uint8_t *src, uint16_t *dst, int width);

#endif // CORE_INTRINSIC_V2_H
