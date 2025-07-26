
# A BAD IDEA is to use column parallelization, as it will cause cache thrashing, 
# thread management overhead and thus reduce performance.

# V4 Implemented a row parallelization, which is more cache-friendly and can be parallelized by multiple cores.
# 105ms

import numpy as np
from numba import jit, prange

# This function remains outside JIT as it uses NumPy features not fully supported by Numba's AOT compilation
def create_bt709_lut(size=65536):
    """
    Creates a lookup table (LUT) for BT.709 Gamma correction.
    """
    linear_input = np.linspace(0, 1, size, dtype=np.float32)
    gamma_corrected_output = np.where(linear_input < 0.018,
                                      4.5 * linear_input,
                                      1.099 * (linear_input ** 0.45) - 0.099)
    return gamma_corrected_output.astype(np.float32)

BT709_LUT = create_bt709_lut()

@jit(nopython=True, fastmath=True)
def jit_get_padded_pixel_value(img, black_level, r_padded, c_padded, H_orig, W_orig):
    """
    Gets a pixel value from the original image, applying black level correction and reflect padding logic.
    r_padded, c_padded are the logical coordinates in the padded image.
    """
    r_orig = r_padded - 1
    c_orig = c_padded - 1

    # Apply reflect padding logic
    if r_orig < 0:
        r_orig = -r_orig
    elif r_orig >= H_orig:
        r_orig = H_orig - 2 - (r_orig - H_orig)

    if c_orig < 0:
        c_orig = -c_orig
    elif c_orig >= W_orig:
        c_orig = W_orig - 2 - (c_orig - W_orig)

    val = img[r_orig, c_orig] - black_level
    if val < 0:
        val = 0
    return float(val)

@jit(nopython=True, fastmath=True)
def jit_white_balance_pixel(pixel_val, r, c, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level):
    """
    Performs white balance for a single pixel.
    """
    is_row_even = ((r - 1) % 2 == 0)
    is_col_even = ((c - 1) % 2 == 0)
    
    if pattern_is_bggr:
        if is_row_even and is_col_even: # Blue
            pixel_val = (pixel_val - b_dBLC) * b_gain
        elif not is_row_even and not is_col_even: # Red
            pixel_val = (pixel_val - r_dBLC) * r_gain
        else: # Green
            pixel_val = (pixel_val - g_dBLC) * g_gain
    else: # RGGB
        if is_row_even and is_col_even: # Red
            pixel_val = (pixel_val - r_dBLC) * r_gain
        elif not is_row_even and not is_col_even: # Blue
            pixel_val = (pixel_val - b_dBLC) * b_gain
        else: # Green
            pixel_val = (pixel_val - g_dBLC) * g_gain

    if pixel_val < 0:
        pixel_val = 0
    if pixel_val > clip_max_level:
        pixel_val = clip_max_level
    
    return pixel_val

@jit(nopython=True, fastmath=True, parallel=True) # Added parallel=True
def jit_full_pipeline(img, black_level, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level, conversion_mtx, gamma_lut):
    """
    Processes a Bayer image to an RGB image using a fully fused, JIT-compiled pipeline with on-the-fly padding.
    """
    H_orig, W_orig = img.shape
    H_padded, W_padded = H_orig + 2, W_orig + 2 # H_padded, W_padded refer to the logical padded dimensions
    h, w = H_orig, W_orig # h, w refer to the final output image dimensions
    final_img = np.empty((h, w, 3), dtype=np.float32)
    lut_max_index = len(gamma_lut) - 1

    # Allocate three buffers for prev, curr, next lines for circular buffer (simulated by a list)
    # These buffers will store white-balanced pixel values
    line_buffers = [np.empty(W_padded, dtype=np.float32) for _ in range(3)]

    # Initial fill of the first two line buffers (corresponding to padded_img rows 0 and 1)
    # line_buffers[0] will be prev (logical padded row 0)
    # line_buffers[1] will be curr (logical padded row 1)
    for c_padded in prange(W_padded): # Changed to prange
        line_buffers[0][c_padded] = jit_white_balance_pixel(
            jit_get_padded_pixel_value(img, black_level, 0, c_padded, H_orig, W_orig),
            0, c_padded, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level
        )
    for c_padded in prange(W_padded): # Changed to prange
        line_buffers[1][c_padded] = jit_white_balance_pixel(
            jit_get_padded_pixel_value(img, black_level, 1, c_padded, H_orig, W_orig),
            1, c_padded, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level
        )

    # Main loop processing row by row (r_padded corresponds to the logical padded row index)
    # The loop runs from logical padded row 1 to H_padded - 2 (inclusive)
    # The output image row is r_padded - 1
    for r_padded in range(1, H_padded - 1):
        # Calculate indices for prev, curr, next lines in the circular buffer
        prev_idx = (r_padded - 1) % 3
        curr_idx = r_padded % 3
        next_idx = (r_padded + 1) % 3

        wb_line_prev = line_buffers[prev_idx]
        wb_line_curr = line_buffers[curr_idx]
        wb_line_next = line_buffers[next_idx]

        # Pre-calculate the next white-balanced line into wb_line_next buffer
        # This corresponds to logical padded row r_padded + 1
        for c_padded in prange(W_padded): # Changed to prange
            wb_line_next[c_padded] = jit_white_balance_pixel(
                jit_get_padded_pixel_value(img, black_level, r_padded + 1, c_padded, H_orig, W_orig),
                r_padded + 1, c_padded, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level
            )

        # Process columns for the current output row (r_padded - 1)
        # c_padded corresponds to the logical padded column index
        for c_padded in prange(1, W_padded - 1): # Changed to prange
            # --- Debayer ---
            r_val, g_val, b_val = 0.0, 0.0, 0.0
            
            # Use original image index for pattern
            is_row_even = ((r_padded - 1) % 2 == 0)
            is_col_even = ((c_padded - 1) % 2 == 0)

            if pattern_is_bggr:
                if is_row_even and is_col_even: # Blue
                    b_val = wb_line_curr[c_padded]
                    g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / 4.0
                    r_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / 4.0
                elif is_row_even and not is_col_even: # Green
                    g_val = wb_line_curr[c_padded]
                    b_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / 2.0
                    r_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / 2.0
                elif not is_row_even and is_col_even: # Green
                    g_val = wb_line_curr[c_padded]
                    r_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / 2.0
                    b_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / 2.0
                else: # Red
                    r_val = wb_line_curr[c_padded]
                    g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / 4.0
                    b_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / 4.0
            else: # RGGB
                if is_row_even and is_col_even: # Red
                    r_val = wb_line_curr[c_padded]
                    g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / 4.0
                    b_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / 4.0
                elif is_row_even and not is_col_even: # Green
                    g_val = wb_line_curr[c_padded]
                    r_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / 2.0
                    b_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / 2.0
                elif not is_row_even and is_col_even: # Green
                    g_val = wb_line_curr[c_padded]
                    b_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / 2.0
                    r_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / 2.0
                else: # Blue
                    b_val = wb_line_curr[c_padded]
                    g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / 4.0
                    r_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / 4.0
            
            # Normalize to [0, 1]
            r_norm = r_val / clip_max_level
            g_norm = g_val / clip_max_level
            b_norm = b_val / clip_max_level

            # --- Color Transform ---
            r_ccm = r_norm * conversion_mtx[0, 0] + g_norm * conversion_mtx[0, 1] + b_norm * conversion_mtx[0, 2]
            g_ccm = r_norm * conversion_mtx[1, 0] + g_norm * conversion_mtx[1, 1] + b_norm * conversion_mtx[1, 2]
            b_ccm = r_norm * conversion_mtx[2, 0] + g_norm * conversion_mtx[2, 1] + b_norm * conversion_mtx[2, 2]

            # --- Gamma Correction ---
            if r_ccm < 0.0: r_ccm = 0.0
            if r_ccm > 1.0: r_ccm = 1.0
            if g_ccm < 0.0: g_ccm = 0.0
            if g_ccm > 1.0: g_ccm = 1.0
            if b_ccm < 0.0: b_ccm = 0.0
            if b_ccm > 1.0: b_ccm = 1.0

            r_idx = int(round(r_ccm * lut_max_index))
            g_idx = int(round(g_ccm * lut_max_index))
            b_idx = int(round(b_ccm * lut_max_index))

            final_img[r_padded-1, c_padded-1, 0] = gamma_lut[r_idx]
            final_img[r_padded-1, c_padded-1, 1] = gamma_lut[g_idx]
            final_img[r_padded-1, c_padded-1, 2] = gamma_lut[b_idx]

    return final_img

@jit(nopython=True, fastmath=True)
def raw_processing_jit_V3_1(img: np.ndarray, 
                         black_level: int,
                         ADC_max_level: int,
                         bayer_pattern: str, 
                         wb_params: tuple, 
                         fwd_mtx: np.ndarray, 
                         render_mtx: np.ndarray,
                         gamma: str = 'BT709',
                         ) -> np.ndarray:
    """
    Processes a Bayer RAW image to an RGB image using the fully fused V5 JIT pipeline with on-the-fly padding and column-wise parallelism.
    """
    
    # 1. Prepare parameters
    r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC = wb_params
    pattern_is_bggr = (bayer_pattern == 'BGGR')
    clip_max_level = float(ADC_max_level - black_level)
    conversion_mtx = np.dot(render_mtx, fwd_mtx)
    
    # 2. Call the fully fused JIT pipeline
    if gamma == 'BT709':
        final_float = jit_full_pipeline(img, black_level, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level, conversion_mtx, BT709_LUT)
    else:
        # Fallback for non-BT709 gamma is not implemented in the fused JIT function
        # You can add it here if needed, or raise an error.
        raise NotImplementedError("Only BT709 gamma is supported in the JIT V5 pipeline.")

    return final_float