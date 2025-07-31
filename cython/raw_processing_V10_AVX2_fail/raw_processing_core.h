#ifndef RAW_PROCESSING_CORE_H
#define RAW_PROCESSING_CORE_H

#include <stdint.h> // For uint16_t

// Enum to define indices for the timing results array
typedef enum {
    TIMING_PREPARE_BUFFER = 0,
    TIMING_DEBAYER = 1,
    TIMING_CCM_WB = 2,
    TIMING_GAMMA = 3,
    TIMING_WRITEMEM = 4,
    TIMING_COUNT // Represents the total number of timed sections
} TimingSections;

// Define a boolean type for C, if not using C99's <stdbool.h>
// This makes the code compatible with older C standards.
#ifndef __cplusplus
    #ifndef bool
        #define bool int
        #define true 1
        #define false 0
    #endif
#endif


/**
 * @brief Processes a Bayer image to an RGB image using a pure C pipeline.
 *
 * This function encapsulates the entire image processing pipeline, including
 * white balance, debayering, color correction, and gamma correction.
 * It is designed to be called from a Cython wrapper.
 *
 * @param img Pointer to the input Bayer image data (uint16_t).
 * @param H_orig Original height of the image.
 * @param W_orig Original width of the image.
 * @param black_level The black level to be subtracted from pixel values.
 * @param r_gain Red channel gain for white balance.
 * @param g_gain Green channel gain for white balance.
 * @param b_gain Blue channel gain for white balance.
 * @param r_dBLC Red channel digital black level compensation.
 * @param g_dBLC Green channel digital black level compensation.
 * @param b_dBLC Blue channel digital black level compensation.
 * @param pattern_is_bggr Boolean flag, true if the Bayer pattern is BGGR, false for RGGB.
 * @param clip_max_level The maximum pixel value after black level subtraction.
 * @param conversion_mtx Pointer to a 3x3 color correction matrix (row-major).
 * @param gamma_lut Pointer to the gamma lookup table.
 * @param gamma_lut_size The number of elements in the gamma LUT.
 * @param final_img Pointer to the output RGB image buffer (float, HxWx3).
 * @param line_buffers Pointer to a pre-allocated buffer for white-balanced lines.
 * @param rgb_line_buffer Pointer to a pre-allocated buffer for debayered RGB lines.
 * @param ccm_line_buffer Pointer to a pre-allocated buffer for CCM-processed lines.
 */
void c_full_pipeline(
    const uint16_t* restrict img,
    int H_orig,
    int W_orig,
    int black_level,
    float r_gain, float g_gain, float b_gain,
    float r_dBLC, float g_dBLC, float b_dBLC,
    bool pattern_is_bggr,
    float clip_max_level,
    const float* restrict conversion_mtx,
    const float* restrict gamma_lut,
    int gamma_lut_size,
    float* restrict final_img,
    float* restrict line_buffers,
    float* restrict rgb_line_buffer,
    int* restrict ccm_line_buffer,
    long long* restrict timing_results // Array of size TIMING_COUNT
);

#endif // RAW_PROCESSING_CORE_H
