#include "raw_processing_core.h"
#include <math.h>   // For powf
#include <stddef.h> // For size_t

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
