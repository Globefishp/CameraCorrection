#include "raw_processing_core.h"
#include <math.h>   // For powf
#include <stddef.h> // For size_t
#include <immintrin.h> // For AVX intrinsics

// 12.868 ± 0.253 ms

// Prepare Buffer time: 0.647 ms
// Debayer time: 4.147 ms
// CCM & WB time: 2.364 ms
// Gamma + Wirte Mem time: 4.739 ms
// Total C Pipeline time: 11.988 ms

// Forward declaration of helper functions
static inline void prepare_line_buffer(
    const uint16_t* restrict img, int H_orig, int W_orig,
    int r_padded,
    float* out_line_buffer);

static inline void debayer_pixel(
    const float* restrict wb_line_prev, const float* restrict wb_line_curr, const float* restrict wb_line_next,
    int c_padded_inner, bool is_row_even, bool is_col_even, bool pattern_is_bggr,
    float* restrict r_val, float* restrict g_val, float* restrict b_val);

// Transposes 3x8 (24) floating-point numbers from SoA to AoS format using AVX intrinsics.
static inline void transpose_and_store_8_pixels_avx(const float* r_in, const float* g_in, const float* b_in, float* out_p) {
    // Load 8 floats (256 bits) from each of the R, G, B channels (SoA)
    // Using 'loadu' for unaligned memory access, which is safer.
    __m256 x = _mm256_loadu_ps(r_in);
    __m256 y = _mm256_loadu_ps(g_in);
    __m256 z = _mm256_loadu_ps(b_in);

    // Transpose logic from Intel's example for 3x8 SoA to AoS conversion.
    __m128 *m = (__m128*) out_p;
    __m256 rxy = _mm256_shuffle_ps(x,y, _MM_SHUFFLE(2,0,2,0));
    __m256 ryz = _mm256_shuffle_ps(y,z, _MM_SHUFFLE(3,1,3,1));
    __m256 rzx = _mm256_shuffle_ps(z,x, _MM_SHUFFLE(3,1,2,0));
    __m256 r03 = _mm256_shuffle_ps(rxy, rzx, _MM_SHUFFLE(2,0,2,0));
    __m256 r14 = _mm256_shuffle_ps(ryz, rxy, _MM_SHUFFLE(3,1,2,0));
    __m256 r25 = _mm256_shuffle_ps(rzx, ryz, _MM_SHUFFLE(3,1,3,1));
    
    // Store the transposed 8x3 RGB pixels (AoS) into the output memory.
    m[0] = _mm256_castps256_ps128( r03 );
    m[1] = _mm256_castps256_ps128( r14 );
    m[2] = _mm256_castps256_ps128( r25 );
    m[3] = _mm256_extractf128_ps( r03 ,1);
    m[4] = _mm256_extractf128_ps( r14 ,1);
    m[5] = _mm256_extractf128_ps( r25 ,1);
}

static inline void prepare_line_buffer(
    const uint16_t* restrict img, int H_orig, int W_orig,
    int r_padded,
    float* out_line_buffer)
{
    const int W_padded = W_orig + 2;
    int r_ori_idx;

    // Fill the buffer and pad two ends by reflect 1 pixel.
    if (r_padded == 0) {
        r_ori_idx = 1;
    } else if (r_padded > H_orig - 1) {
        r_ori_idx = H_orig - 2;
    } else {
        r_ori_idx = r_padded - 1;
    }

    const uint16_t* img_row = &img[r_ori_idx * W_orig];
    for (int c_orig = 0; c_orig < W_orig; ++c_orig) {
        out_line_buffer[c_orig + 1] = (float)(img_row[c_orig]);
    }

    out_line_buffer[0] = (float)(img_row[1]);
    out_line_buffer[W_padded - 1] = (float)(img_row[W_orig - 3]);
}

static inline void debayer_pixel(
    const float* restrict wb_line_prev, const float* restrict wb_line_curr, const float* restrict wb_line_next,
    int c_padded_inner, bool is_row_even, bool is_col_even, bool pattern_is_bggr,
    float* restrict r_val, float* restrict g_val, float* restrict b_val)
{
    if (pattern_is_bggr) {
        if (is_row_even && is_col_even) { // Blue
            *b_val = wb_line_curr[c_padded_inner];
            *g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * 0.25f;
            *r_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) * 0.25f;
        } else if (is_row_even && !is_col_even) { // Green
            *g_val = wb_line_curr[c_padded_inner];
            *b_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) * 0.5f;
            *r_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * 0.5f;
        } else if (!is_row_even && is_col_even) { // Green
            *g_val = wb_line_curr[c_padded_inner];
            *r_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) * 0.5f;
            *b_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * 0.5f;
        } else { // Red
            *r_val = wb_line_curr[c_padded_inner];
            *g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * 0.25f;
            *b_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) * 0.25f;
        }
    } else { // RGGB
        if (is_row_even && is_col_even) { // Red
            *r_val = wb_line_curr[c_padded_inner];
            *g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * 0.25f;
            *b_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) * 0.25f;
        } else if (is_row_even && !is_col_even) { // Green
            *g_val = wb_line_curr[c_padded_inner];
            *r_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) * 0.5f;
            *b_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * 0.5f;
        } else if (!is_row_even && is_col_even) { // Green
            *g_val = wb_line_curr[c_padded_inner];
            *b_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) * 0.5f;
            *r_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * 0.5f;
        } else { // Blue
            *b_val = wb_line_curr[c_padded_inner];
            *g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * 0.25f;
            *r_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) * 0.25f;
        }
    }
}

void c_full_pipeline(
    const uint16_t* restrict img, int H_orig, int W_orig, int black_level,
    float r_gain, float g_gain, float b_gain, float r_dBLC, float g_dBLC, float b_dBLC,
    bool pattern_is_bggr, float clip_max_level,
    const float* restrict conversion_mtx, const uint16_t* restrict gamma_lut, int gamma_lut_size,
    uint16_t* restrict final_img, float* restrict line_buffers, float* restrict rgb_line_buffer, int* restrict ccm_line_buffer)
{
    const int H_padded = H_orig + 2;
    const int W_padded = W_orig + 2;
    const int lut_max_index = gamma_lut_size - 1;
    const float inv_clip_max_level = 1.0f / clip_max_level;

    float r_BLC = r_dBLC + black_level;
    float g_BLC = g_dBLC + black_level;
    float b_BLC = b_dBLC + black_level;

    float* r_line_buffer = &rgb_line_buffer[0 * W_orig];
    float* g_line_buffer = &rgb_line_buffer[1 * W_orig];
    float* b_line_buffer = &rgb_line_buffer[2 * W_orig];

    // Using separate buffers give more optimization opportunity for compiler.
    int* r_ccm_line = &ccm_line_buffer[0 * W_orig];
    int* g_ccm_line = &ccm_line_buffer[1 * W_orig];
    int* b_ccm_line = &ccm_line_buffer[2 * W_orig];

    const float m00 = conversion_mtx[0], m01 = conversion_mtx[1], m02 = conversion_mtx[2];
    const float m10 = conversion_mtx[3], m11 = conversion_mtx[4], m12 = conversion_mtx[5];
    const float m20 = conversion_mtx[6], m21 = conversion_mtx[7], m22 = conversion_mtx[8];

    // Pre-fill the first two line buffers
    prepare_line_buffer(img, H_orig, W_orig, 0, &line_buffers[0 * W_padded]);
    prepare_line_buffer(img, H_orig, W_orig, 1, &line_buffers[1 * W_padded]);

    for (int r_padded = 1; r_padded < H_padded - 1; ++r_padded) {
        const int prev_idx = (r_padded - 1) % 3;
        const int curr_idx = r_padded % 3;
        const int next_idx = (r_padded + 1) % 3;

        float* wb_line_prev = &line_buffers[prev_idx * W_padded];
        float* wb_line_curr = &line_buffers[curr_idx * W_padded];
        float* wb_line_next = &line_buffers[next_idx * W_padded];

        prepare_line_buffer(img, H_orig, W_orig, r_padded + 1, wb_line_next);

        // Cache-intensive, no vectorization.
        for (int c_padded_inner = 1; c_padded_inner < W_padded - 1; ++c_padded_inner) {
            const int c_orig_inner = c_padded_inner - 1;
            const bool is_row_even = ((r_padded - 1) % 2 == 0);
            const bool is_col_even = ((c_padded_inner - 1) % 2 == 0);
            
            float r_val, g_val, b_val;
            debayer_pixel(wb_line_prev, wb_line_curr, wb_line_next, c_padded_inner, is_row_even, is_col_even, pattern_is_bggr, &r_val, &g_val, &b_val);
            
            r_line_buffer[c_orig_inner] = r_val;
            g_line_buffer[c_orig_inner] = g_val;
            b_line_buffer[c_orig_inner] = b_val;
        }

        // Combine WB, clipping and CCM loops in one compute-intensive cycle.
        // Vectorized by compiler
        for (int c = 0; c < W_orig; ++c) {
            float r_in = r_line_buffer[c];
            float g_in = g_line_buffer[c];
            float b_in = b_line_buffer[c];
            
            // White-balance in native range
            r_in = (r_in - r_BLC) * r_gain ;
            g_in = (g_in - g_BLC) * g_gain ;
            b_in = (b_in - b_BLC) * b_gain ;
            // Clip + Normalize
            r_in = (r_in < clip_max_level) ? r_in * inv_clip_max_level : 1.0f;
            g_in = (g_in < clip_max_level) ? g_in * inv_clip_max_level : 1.0f;
            b_in = (b_in < clip_max_level) ? b_in * inv_clip_max_level : 1.0f;
            // Prevent underflow.
            r_in = (r_in > 0.0f) ? r_in : 0.0f;
            g_in = (g_in > 0.0f) ? g_in : 0.0f;
            b_in = (b_in > 0.0f) ? b_in : 0.0f;
            
            // CCM
            // R channel 
            float val_r = r_in * m00 + g_in * m01 + b_in * m02;
            val_r = (val_r < 1.0f) ? val_r : 1.0f;
            val_r = (val_r > 0.0f) ? val_r : 0.0f;
            r_ccm_line[c] = (int)(val_r * lut_max_index + 0.5f);
            
            // G channel
            float val_g = r_in * m10 + g_in * m11 + b_in * m12;
            val_g = (val_g < 1.0f) ? val_g : 1.0f;
            val_g = (val_g > 0.0f) ? val_g : 0.0f;
            g_ccm_line[c] = (int)(val_g * lut_max_index + 0.5f);

            // B channel
            float val_b = r_in * m20 + g_in * m21 + b_in * m22;
            val_b = (val_b < 1.0f) ? val_b : 1.0f;
            val_b = (val_b > 0.0f) ? val_b : 0.0f;
            b_ccm_line[c] = (int)(val_b * lut_max_index + 0.5f);
        }

        uint16_t* out_row = &final_img[(r_padded - 1) * W_orig * 3];
        // Can be unrolled by compiler to ultilize all register.
        // Bottleneck: write back to memory (or SoA->AoS). Gamma time only 1.4ms @ 65536 size.
        // Can be check by set index to 0 to eliminate random access.
        // If separate into LUT loop and write loop: LUT loop: 3.6ms, write loop still ~8ms.
        // Thus we can see the cost of accessing the buffer line: in AVX2 access (Prepare buffer: 0.7ms); in scaler access: 2.2ms.
        // The stubborn 8ms is likely cause by write back. Not SoA->AoS (AVX2 intrinsic by Intel gives the same speed.) 
        // See AVX2 version timing.
        // TODO: reduce output to 16bit int. using aligned cache line.
        for (int c = 0; c < W_orig; ++c) {
            out_row[c * 3 + 0] = gamma_lut[r_ccm_line[c]];
            out_row[c * 3 + 1] = gamma_lut[g_ccm_line[c]];
            out_row[c * 3 + 2] = gamma_lut[b_ccm_line[c]];
        }
    }
}
