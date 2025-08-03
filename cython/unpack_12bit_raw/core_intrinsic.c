#include <immintrin.h>
#include <stdint.h>

// Using CPU capabilities: SSE(_mm_shuffle_ps), SSE2(_mm_loadu/storeu_si128, _mm_srli_epi16, _mm_andnot/and/or_si128), SSSE3(_mm_shuffle_epi8), 

// SIMD implementation (memory bound): 2.569 ± 0.985 ms, n=1000. 
// Read: 2448*2048*1.5Bpp=7.52 MB, Write: 2448*2048*2Bpp=10.02 MB. 
// Total throughput: 6.8GB/s

// Naive C implementation:
//   4.027 ± 1.398 ms, n=1000.


// Forward declaration for the main function
void unpack_12bit_raw(uint8_t *src, uint16_t *dst, int width);

static inline void unpack_12bit_raw_xmm(uint8_t *src, uint16_t *dst)
{
    // Using 128bit SIMD. In 256 bit implementation, we need vinsertf128, 
    //   which may be too slow and expensive (Latency 3, CPI 1)for this simple task.
    // By calculation, the core intrinsics latency is 11 (critical path, including loading 6), 4 step calc included.
    // Data from Intel Intrinsic Guide: CPI is based on Alderlake, 
    // for shuffle instruction, the CPI on Skylake is 1. All intrinsics have latency 1 except for load.
    // search uops.info for more instructions.

    // shuffle_ps(CPI 0.5) is used to reorder the 3 x 128bit(xmm) input data for twice(CPI0.5), creating two temp registers.
    // 2 input xmm + 2 temp xmm -> 4 output xmm(4 groups, each contain 4 pair of packed pixels), the 12->16bit extended here.
    // The following step is to make the value within each xmm right.
    // The packed rule for two pixels is: 
    //     byte0 and byte2 is the high 8 bit of pixel0 and pixel1.
    //     byte1 low 4 bit is pixel0 low 4 bit.
    //     byte1 high 4 bit is pixel1 low 4 bit.
    // Thus the following steps to manipulate 4 output registers including:
    // For each xmm (total 4):
    //   shuffle_epi8 (CPI 0.5) once (copy byte1 to byte3 to concatenate with pixel1 high byte(byte2), thus each 2-pixel group is now 32 bit)
    //   branch 1:
    //     shift right logical immediate (slli_epi16, CPI 0.5) to make correct 12 bit range.
    //     mask out (and_si128, CPI 0.333) the low 4 bit(which actually belongs to pixel1) of pixel0 in each pixel group.
    //   branch 2:
    //     mask to get only low 4 bit of byte 1 (and_si128, CPI 0.333) thoughout the register.
    //   blend result:
    //     bitwise or (or_si128, CPI 0.333) of branch 1 and branch 2.

    // load data (Latency 6, CPI 0.333)
    __m128i input0 = _mm_loadu_si128((__m128i const*) src); // group0 
    __m128i input1 = _mm_loadu_si128((__m128i const*)(src + 16)); // just a temp input
    __m128i input2 = _mm_loadu_si128((__m128i const*)(src + 32)); // group3

    __m128i group1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(input0), _mm_castsi128_ps(input1), _MM_SHUFFLE(1, 0, 3, 2))); 
    __m128i group2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(input1), _mm_castsi128_ps(input2), _MM_SHUFFLE(1, 0, 3, 2))); 

    // shuffle within the xmm to get 16 bit format. the pattern of group0/2 is the same(high 96 bits effective). group1/3 is the same(low 96 bits effective).
    // Remember: 16bit is LITTLE_ENDIAN.
    __m128i group0_shuf8 = _mm_shuffle_epi8(input0, _mm_setr_epi8(1, 0, 1, 2, 4, 3, 4, 5, 7, 6, 7, 8, 10, 9, 10, 11));
    __m128i group2_shuf8 = _mm_shuffle_epi8(group2, _mm_setr_epi8(1, 0, 1, 2, 4, 3, 4, 5, 7, 6, 7, 8, 10, 9, 10, 11));

    __m128i group1_shuf8 = _mm_shuffle_epi8(group1, _mm_setr_epi8(5, 4, 5, 6, 8, 7, 8, 9, 11, 10, 11, 12, 14, 13, 14, 15));
    __m128i group3_shuf8 = _mm_shuffle_epi8(input2, _mm_setr_epi8(5, 4, 5, 6, 8, 7, 8, 9, 11, 10, 11, 12, 14, 13, 14, 15));

//     // debug: mask all low 8 bit, if you use epi32 mask, remember the literal high 16 bit is actually in the right in your imageination (high address)
//     // Thus it's better to use epi16
//             group0_shuf8 = _mm_andnot_si128(_mm_setr_epi32(0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF), group0_shuf8);
//             group1_shuf8 = _mm_andnot_si128(_mm_setr_epi32(0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF), group1_shuf8);
//             group2_shuf8 = _mm_andnot_si128(_mm_setr_epi32(0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF), group2_shuf8);
//             group3_shuf8 = _mm_andnot_si128(_mm_setr_epi32(0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF), group3_shuf8);

    // branch 1
    __m128i group0_b1 = _mm_srli_epi16(group0_shuf8, 4);
    __m128i group1_b1 = _mm_srli_epi16(group1_shuf8, 4);
    __m128i group2_b1 = _mm_srli_epi16(group2_shuf8, 4);
    __m128i group3_b1 = _mm_srli_epi16(group3_shuf8, 4);

            group0_b1 = _mm_andnot_si128(           _mm_setr_epi16(0x000F, 0, 0x000F, 0, 0x000F, 0, 0x000F, 0), group0_b1);
            group1_b1 = _mm_andnot_si128(           _mm_setr_epi16(0x000F, 0, 0x000F, 0, 0x000F, 0, 0x000F, 0), group1_b1);
            group2_b1 = _mm_andnot_si128(           _mm_setr_epi16(0x000F, 0, 0x000F, 0, 0x000F, 0, 0x000F, 0), group2_b1);
            group3_b1 = _mm_andnot_si128(           _mm_setr_epi16(0x000F, 0, 0x000F, 0, 0x000F, 0, 0x000F, 0), group3_b1);


    // branch 2
    __m128i group0_b2 = _mm_and_si128(group0_shuf8, _mm_setr_epi16(0x000F, 0, 0x000F, 0, 0x000F, 0, 0x000F, 0));
    __m128i group1_b2 = _mm_and_si128(group1_shuf8, _mm_setr_epi16(0x000F, 0, 0x000F, 0, 0x000F, 0, 0x000F, 0));
    __m128i group2_b2 = _mm_and_si128(group2_shuf8, _mm_setr_epi16(0x000F, 0, 0x000F, 0, 0x000F, 0, 0x000F, 0));
    __m128i group3_b2 = _mm_and_si128(group3_shuf8, _mm_setr_epi16(0x000F, 0, 0x000F, 0, 0x000F, 0, 0x000F, 0));

    // blend
    __m128i out0 = _mm_or_si128(group0_b1, group0_b2);
    __m128i out1 = _mm_or_si128(group1_b1, group1_b2);
    __m128i out2 = _mm_or_si128(group2_b1, group2_b2);
    __m128i out3 = _mm_or_si128(group3_b1, group3_b2);

    // Store results
    _mm_storeu_si128((__m128i*)(dst)     , out0);
    _mm_storeu_si128((__m128i*)(dst + 8) , out1);
    _mm_storeu_si128((__m128i*)(dst + 16), out2);
    _mm_storeu_si128((__m128i*)(dst + 24), out3);
}

/**
 * @brief Unpacks a 12-bit RAW image buffer to a 16-bit buffer using SIMD instructions.
 * 
 * This function processes the image data in chunks of 32 pixels (48 bytes) for optimal
 * performance. Any remaining pixels at the end are handled by a standard C loop.
 * 
 * @param src Pointer to the source 12-bit RAW data buffer.
 * @param dst Pointer to the destination 16-bit buffer.
 * @param width The number of pixels to process. 
 *              NOTE: It is the caller's responsibility to ensure that the width is an even number.
 */
void unpack_12bit_raw(uint8_t *src, uint16_t *dst, int width)
{
    int i = 0;
    int main_loop_limit = width - (width % 32);

    // Main loop: process data in chunks of 32 pixels (48 bytes src -> 64 bytes dst)
    for (i = 0; i < main_loop_limit; i += 32) {
        unpack_12bit_raw_xmm(src, dst);
        src += 48; // 32 pixels * 1.5 bytes/pixel
        dst += 32; // 32 pixels * 2 bytes/pixel
    }

    // Handle any remaining pixels that are not a multiple of 32
    int remaining_pixels = width - main_loop_limit;
    if (remaining_pixels > 0) {
        for (i = 0; i < remaining_pixels / 2; ++i) {
            uint8_t byte0 = src[i * 3];
            uint8_t byte1 = src[i * 3 + 1];
            uint8_t byte2 = src[i * 3 + 2];

            dst[i * 2] = (uint16_t)(byte0 << 4) | (uint16_t)(byte1 & 0x0F);
            dst[i * 2 + 1] = (uint16_t)(byte2 << 4) | (uint16_t)(byte1 >> 4);
        }
    }

    // // Naive C implementation for compiler auto-vectorization test
    // for (int i = 0; i < width / 2; ++i) {
    //     uint8_t byte0 = src[i * 3];
    //     uint8_t byte1 = src[i * 3 + 1];
    //     uint8_t byte2 = src[i * 3 + 2];

    //     dst[i * 2] = (uint16_t)(byte0 << 4) | (uint16_t)(byte1 & 0x0F);
    //     dst[i * 2 + 1] = (uint16_t)(byte2 << 4) | (uint16_t)(byte1 >> 4);
    // }
}
