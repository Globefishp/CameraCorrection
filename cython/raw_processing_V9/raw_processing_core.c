#include "raw_processing_core.h"
#include <math.h>   // For powf
#include <stddef.h> // For size_t
#include <immintrin.h> // For AVX, AVX2

// 17.354 ± 0.400 ms

// Forward declaration of helper functions
static inline void white_balance_line_vectorized(
    const uint16_t* restrict img, int H_orig, int W_orig, int black_level,
    int r_padded, float r_gain, float g_gain, float b_gain,
    float r_dBLC, float g_dBLC, float b_dBLC,
    bool pattern_is_bggr, float clip_max_level,
    float* out_line_buffer);

static inline void debayer_pixel(
    const float* restrict wb_line_prev, const float* restrict wb_line_curr, const float* restrict wb_line_next,
    int c_padded_inner, bool is_row_even, bool is_col_even, bool pattern_is_bggr,
    float* restrict r_val, float* restrict g_val, float* restrict b_val);

// Forward declaration for AVX2-ready functions
static inline void debayer_bggr_even_row_avx2(
    const float* restrict wb_line_prev, const float* restrict wb_line_curr, const float* restrict wb_line_next,
    int c_padded_inner,
    float* restrict r_out, float* restrict g_out, float* restrict b_out);

static inline void debayer_bggr_odd_row_avx2(
    const float* restrict wb_line_prev, const float* restrict wb_line_curr, const float* restrict wb_line_next,
    int c_padded_inner,
    float* restrict r_out, float* restrict g_out, float* restrict b_out);


static inline void white_balance_line_vectorized(
    const uint16_t* restrict img, int H_orig, int W_orig, int black_level,
    int r_padded, float r_gain, float g_gain, float b_gain,
    float r_dBLC, float g_dBLC, float b_dBLC,
    bool pattern_is_bggr, float clip_max_level,
    float* out_line_buffer)
{
    const int W_padded = W_orig + 2;
    int r_ori_idx;

    // Step 1: Fill the buffer with black-level corrected values.
    if (r_padded == 0) {
        r_ori_idx = 1;
    } else if (r_padded > H_orig - 1) {
        r_ori_idx = H_orig - 2;
    } else {
        r_ori_idx = r_padded - 1;
    }

    const uint16_t* img_row = &img[r_ori_idx * W_orig];
    for (int c_orig = 0; c_orig < W_orig; ++c_orig) {
        out_line_buffer[c_orig + 1] = (float)(img_row[c_orig] - black_level);
    }

    out_line_buffer[0] = (float)(img_row[1] - black_level);
    out_line_buffer[W_padded - 1] = (float)(img_row[W_orig - 3] - black_level);

    // Step 2: Apply white balance in-place.
    const bool is_row_even = ((r_padded - 1) % 2 == 0);
    float gain_even, gain_odd, dBLC_even, dBLC_odd;

    if (pattern_is_bggr) {
        if (is_row_even) { // Row is B G B G...
            gain_even = b_gain; dBLC_even = b_dBLC;
            gain_odd = g_gain; dBLC_odd = g_dBLC;
        } else { // Row is G R G R...
            gain_even = g_gain; dBLC_even = g_dBLC;
            gain_odd = r_gain; dBLC_odd = r_dBLC;
        }
    } else { // RGGB
        if (is_row_even) { // Row is R G R G...
            gain_even = r_gain; dBLC_even = r_dBLC;
            gain_odd = g_gain; dBLC_odd = g_dBLC;
        } else { // Row is G B G B...
            gain_even = g_gain; dBLC_even = g_dBLC;
            gain_odd = b_gain; dBLC_odd = b_dBLC;
        }
    }

    const int c_half = W_padded / 2;
    for (int i = 0; i < c_half; ++i) {
        float val1 = (out_line_buffer[i * 2] - dBLC_odd) * gain_odd;
        if (val1 > clip_max_level) val1 = clip_max_level;
        if (val1 < 0.0f) val1 = 0.0f;
        out_line_buffer[i * 2] = val1;
    }
    for (int i = 1; i < c_half; ++i) {
        float val2 = (out_line_buffer[i * 2 + 1] - dBLC_even) * gain_even;
        if (val2 > clip_max_level) val2 = clip_max_level;
        if (val2 < 0.0f) val2 = 0.0f;
        out_line_buffer[i * 2 + 1] = val2;
    }
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

static inline void debayer_bggr_even_row_avx2(
    const float* restrict wb_line_prev, const float* restrict wb_line_curr, const float* restrict wb_line_next,
    int c_padded_inner,
    float* restrict r_out, float* restrict g_out, float* restrict b_out)
{
    // Even row (BGGR): B G B G B G B G
    // Constants for calculations
    const __m256 const_0_5 = _mm256_set1_ps(0.5f);
    const __m256 const_0_25 = _mm256_set1_ps(0.25f);

    // Mask to blend pixels from B-locations (even) and G-locations (odd)
    const __m256 blend_mask = _mm256_castsi256_ps(_mm256_setr_epi32(
        0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
        0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF));

    // Pointers to the start of the 8-pixel block
    const float* p_curr = wb_line_curr + c_padded_inner;
    const float* p_prev = wb_line_prev + c_padded_inner;
    const float* p_next = wb_line_next + c_padded_inner;

    // Load data for current, previous, and next lines
    const __m256 curr_m = _mm256_loadu_ps(p_curr);
    const __m256 prev_m = _mm256_loadu_ps(p_prev);
    const __m256 next_m = _mm256_loadu_ps(p_next);
    const __m256 curr_l = _mm256_loadu_ps(p_curr - 1);
    const __m256 curr_r = _mm256_loadu_ps(p_curr + 1);

    // --- Common Subexpressions ---
    const __m256 curr_lr_sum     = _mm256_add_ps(curr_l, curr_r);
    const __m256 prev_next_m_sum = _mm256_add_ps(prev_m, next_m);

    // --- Green channel calculation ---
    // G at B-pixels (even): (curr[c-1] + curr[c+1] + prev[c] + next[c]) * 0.25
    // G at G-pixels (odd):  curr[c]
    __m256 g_at_b = _mm256_mul_ps(_mm256_add_ps(curr_lr_sum, prev_next_m_sum), const_0_25);
    const __m256 g_out_vec = _mm256_blendv_ps(g_at_b, curr_m, blend_mask);
    // 9 cycle

    // --- Blue channel calculation ---
    // B at B-pixels (even): curr[c]
    // B at G-pixels (odd):  (curr[c-1] + curr[c+1]) * 0.5
    const __m256 b_out_vec = _mm256_blendv_ps(curr_m, _mm256_mul_ps(curr_lr_sum, const_0_5), blend_mask);
    // 7 cycle

    // --- Red channel calculation ---
    // R at B-pixels (even): (prev[c-1] + prev[c+1] + next[c-1] + next[c+1]) * 0.25
    // R at G-pixels (odd):  (prev[c] + next[c]) * 0.5
    const __m256 prev_lr_sum = _mm256_add_ps(_mm256_loadu_ps(p_prev - 1), _mm256_loadu_ps(p_prev + 1));
    const __m256 next_lr_sum = _mm256_add_ps(_mm256_loadu_ps(p_next - 1), _mm256_loadu_ps(p_next + 1));
    __m256 r_at_b = _mm256_fmadd_ps(prev_lr_sum, const_0_25, _mm256_mul_ps(next_lr_sum, const_0_25));
    const __m256 r_out_vec = _mm256_blendv_ps(r_at_b, _mm256_mul_ps(prev_next_m_sum, const_0_5), blend_mask);
    // 11 cycle

    // Store results
    _mm256_storeu_ps(r_out, r_out_vec);
    _mm256_storeu_ps(g_out, g_out_vec);
    _mm256_storeu_ps(b_out, b_out_vec);
}

static inline void debayer_bggr_odd_row_avx2(
    const float* restrict wb_line_prev, const float* restrict wb_line_curr, const float* restrict wb_line_next,
    int c_padded_inner,
    float* restrict r_out, float* restrict g_out, float* restrict b_out)
{
    // Latency and Throughput data on Alderlake (i9-14900k)
    // Odd row (BGGR): G R G R G R G R
    // Constants for calculations
    const __m256 const_0_5 = _mm256_set1_ps(0.5f);
    const __m256 const_0_25 = _mm256_set1_ps(0.25f);

    // Mask to blend pixels from G-locations (even) and R-locations (odd)

    const __m256 blend_mask = _mm256_castsi256_ps(_mm256_setr_epi32(
        0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
        0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF));

    // Pointers to the start of the 8-pixel block
    const float* p_curr = wb_line_curr + c_padded_inner;
    const float* p_prev = wb_line_prev + c_padded_inner;
    const float* p_next = wb_line_next + c_padded_inner;

    // Load data for current, previous, and next lines
    const __m256 curr_m = _mm256_loadu_ps(p_curr);
    const __m256 prev_m = _mm256_loadu_ps(p_prev);
    const __m256 next_m = _mm256_loadu_ps(p_next); // Throughtput 1/3, Latency 7
    const __m256 curr_l = _mm256_loadu_ps(p_curr - 1);
    const __m256 curr_r = _mm256_loadu_ps(p_curr + 1);

    // SubExpression: curr_lr_sum(curr[c-1] + curr[c+1]) and prev_next_m_sum(prev[c] + next[c])
    const __m256 curr_lr_sum     = _mm256_add_ps(curr_l, curr_r);
    const __m256 prev_next_m_sum = _mm256_add_ps(prev_m, next_m);
    
    // --- Green channel calculation ---
    // G at G-pixels: curr[c]
    // G at R-pixels: (curr[c-1] + curr[c+1] + prev[c] + next[c]) * 0.25
    __m256 g_at_r = _mm256_add_ps(curr_lr_sum, prev_next_m_sum);
    g_at_r = _mm256_mul_ps(g_at_r, const_0_25);
    // Blend to get final G channel
    const __m256 g_out_vec = _mm256_blendv_ps(curr_m, g_at_r, blend_mask);

    // --- Red channel calculation ---
    // R at G-pixels: (curr[c-1] + curr[c+1]) * 0.5
    // R at R-pixels: curr[c]
    // Blend to get final R channel
    const __m256 r_out_vec = _mm256_blendv_ps(_mm256_mul_ps(curr_lr_sum, const_0_5), curr_m, blend_mask);

    // --- Blue channel calculation ---
    // B at G-pixels: (prev[c] + next[c]) * 0.5
    // B at R-pixels: (prev[c-1] + prev[c+1] + next[c-1] + next[c+1]) * 0.25
    const __m256 prev_lr_sum = _mm256_add_ps(_mm256_loadu_ps(p_prev - 1), _mm256_loadu_ps(p_prev + 1));
    const __m256 next_lr_sum = _mm256_add_ps(_mm256_loadu_ps(p_next - 1), _mm256_loadu_ps(p_next + 1));
    __m256 b_at_r = _mm256_fmadd_ps(prev_lr_sum, const_0_25, _mm256_mul_ps(next_lr_sum, const_0_25));
    // Shortest critical path (Latency = 7 + 2 + 4 + 4), Throughtput(CPI) for add mul fmadd is 0.5.
    // Blend to get final B channel
    const __m256 b_out_vec = _mm256_blendv_ps(_mm256_mul_ps(prev_next_m_sum, const_0_5), b_at_r, blend_mask);

    // Store results
    _mm256_storeu_ps(r_out, r_out_vec);
    _mm256_storeu_ps(g_out, g_out_vec);
    _mm256_storeu_ps(b_out, b_out_vec);
}

void c_full_pipeline(
    const uint16_t* restrict img, int H_orig, int W_orig, int black_level,
    float r_gain, float g_gain, float b_gain, float r_dBLC, float g_dBLC, float b_dBLC,
    bool pattern_is_bggr, float clip_max_level,
    const float* restrict conversion_mtx, const float* restrict gamma_lut, int gamma_lut_size,
    float* restrict final_img, float* restrict line_buffers, float* restrict rgb_line_buffer, float* restrict ccm_line_buffer)
{

    const int H_padded = H_orig + 2;
    const int W_padded = W_orig + 2;
    const int lut_max_index = gamma_lut_size - 1;
    const float inv_clip_max_level = 1.0f / clip_max_level;

    float* r_line_buffer = &rgb_line_buffer[0 * W_orig];
    float* g_line_buffer = &rgb_line_buffer[1 * W_orig];
    float* b_line_buffer = &rgb_line_buffer[2 * W_orig];

    float* r_ccm_line = &ccm_line_buffer[0 * W_orig];
    float* g_ccm_line = &ccm_line_buffer[1 * W_orig];
    float* b_ccm_line = &ccm_line_buffer[2 * W_orig];

    const float m00 = conversion_mtx[0], m01 = conversion_mtx[1], m02 = conversion_mtx[2];
    const float m10 = conversion_mtx[3], m11 = conversion_mtx[4], m12 = conversion_mtx[5];
    const float m20 = conversion_mtx[6], m21 = conversion_mtx[7], m22 = conversion_mtx[8];

    // Pre-fill the first two line buffers
    white_balance_line_vectorized(img, H_orig, W_orig, black_level, 0, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level, &line_buffers[0 * W_padded]);
    white_balance_line_vectorized(img, H_orig, W_orig, black_level, 1, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level, &line_buffers[1 * W_padded]);

    for (int r_padded = 1; r_padded < H_padded - 1; ++r_padded) {
        const int prev_idx = (r_padded - 1) % 3;
        const int curr_idx = r_padded % 3;
        const int next_idx = (r_padded + 1) % 3;

        float* wb_line_prev = &line_buffers[prev_idx * W_padded];
        float* wb_line_curr = &line_buffers[curr_idx * W_padded];
        float* wb_line_next = &line_buffers[next_idx * W_padded];

        white_balance_line_vectorized(img, H_orig, W_orig, black_level, r_padded + 1, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level, wb_line_next);

        const bool is_row_even = ((r_padded - 1) % 2 == 0);
        if (pattern_is_bggr) {
            if (is_row_even) {
                // Process even row (B-G) with 8-pixel steps
                for (int c_padded_inner = 1; c_padded_inner < W_padded - 1; c_padded_inner += 8) {
                    const int c_orig_inner = c_padded_inner - 1;
                    debayer_bggr_even_row_avx2(wb_line_prev, wb_line_curr, wb_line_next, c_padded_inner, &r_line_buffer[c_orig_inner], &g_line_buffer[c_orig_inner], &b_line_buffer[c_orig_inner]);
                }
            } else {
                // Process odd row (G-R) with 8-pixel steps
                for (int c_padded_inner = 1; c_padded_inner < W_padded - 1; c_padded_inner += 8) {
                    const int c_orig_inner = c_padded_inner - 1;
                    debayer_bggr_odd_row_avx2(wb_line_prev, wb_line_curr, wb_line_next, c_padded_inner, &r_line_buffer[c_orig_inner], &g_line_buffer[c_orig_inner], &b_line_buffer[c_orig_inner]);
                }
            }
        } else {
            // RGGB or other patterns not implemented for AVX2 path yet.
            // Fill with zeros as requested.
            for (int c = 0; c < W_orig; ++c) {
                r_line_buffer[c] = 0.0f;
                g_line_buffer[c] = 0.0f;
                b_line_buffer[c] = 0.0f;
            }
        }

        // Combine CCM loops while keeping explicit clamping logic
        for (int c = 0; c < W_orig; ++c) {
            const float r_in = r_line_buffer[c] * inv_clip_max_level;
            const float g_in = g_line_buffer[c] * inv_clip_max_level;
            const float b_in = b_line_buffer[c] * inv_clip_max_level;
        
            // R channel
            float val_r = r_in * m00 + g_in * m01 + b_in * m02;
            val_r = (val_r < 1.0f) ? val_r : 1.0f;
            val_r = (val_r > 0.0f) ? val_r : 0.0f;
            r_ccm_line[c] = val_r * lut_max_index + 0.5f;
            
            // G channel
            float val_g = r_in * m10 + g_in * m11 + b_in * m12;
            val_g = (val_g < 1.0f) ? val_g : 1.0f;
            val_g = (val_g > 0.0f) ? val_g : 0.0f;
            g_ccm_line[c] = val_g * lut_max_index + 0.5f;

            // B channel
            float val_b = r_in * m20 + g_in * m21 + b_in * m22;
            val_b = (val_b < 1.0f) ? val_b : 1.0f;
            val_b = (val_b > 0.0f) ? val_b : 0.0f;
            b_ccm_line[c] = val_b * lut_max_index + 0.5f;
        }

        float* out_row = &final_img[(r_padded - 1) * W_orig * 3];
        for (int c = 0; c < W_orig; ++c) {
            out_row[c * 3 + 0] = gamma_lut[(int)r_ccm_line[c]];
            out_row[c * 3 + 1] = gamma_lut[(int)g_ccm_line[c]];
            out_row[c * 3 + 2] = gamma_lut[(int)b_ccm_line[c]];
        }
    }
}
