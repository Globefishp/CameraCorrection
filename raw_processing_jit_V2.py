# 40.272 ± 1.075 ms

import numpy as np
from numba import jit

# This function remains outside JIT as it uses NumPy features not fully supported by Numba's AOT compilation
def create_bt709_lut(size=65536):
    """
    Creates a lookup table (LUT) for BT.709 Gamma correction.
    """
    linear_input = np.linspace(0, 1, size, dtype=np.float64)
    gamma_corrected_output = np.where(linear_input < 0.018,
                                      4.5 * linear_input,
                                      1.099 * (linear_input ** 0.45) - 0.099)
    return gamma_corrected_output.astype(np.float64)

BT709_LUT = create_bt709_lut()

@jit(nopython=True, fastmath=True)
def jit_prepare_padded_image(img, black_level):
    H_orig, W_orig = img.shape
    H_padded, W_padded = H_orig + 2, W_orig + 2
    padded_img = np.empty((H_padded, W_padded), dtype=np.float64)

    # Fill center (original image data) and apply black level correction + type conversion
    for r in range(H_orig):
        for c in range(W_orig):
            val = img[r, c] - black_level
            if val < 0:
                val = 0
            padded_img[r + 1, c + 1] = float(val)

    # Manually implement reflect padding
    # Top and Bottom rows
    for c in range(1, W_orig + 1):
        padded_img[0, c] = padded_img[2, c] # Reflect row 1 to row 0
        padded_img[H_padded - 1, c] = padded_img[H_padded - 3, c] # Reflect row H_padded-2 to row H_padded-1

    # Left and Right columns
    for r in range(1, H_orig + 1):
        padded_img[r, 0] = padded_img[r, 2] # Reflect col 1 to col 0
        padded_img[r, W_padded - 1] = padded_img[r, W_padded - 3] # Reflect col W_padded-2 to col W_padded-1

    # Corners
    padded_img[0, 0] = padded_img[2, 2] # Top-left
    padded_img[0, W_padded - 1] = padded_img[2, W_padded - 3] # Top-right
    padded_img[H_padded - 1, 0] = padded_img[H_padded - 3, 2] # Bottom-left
    padded_img[H_padded - 1, W_padded - 1] = padded_img[H_padded - 3, W_padded - 3] # Bottom-right

    return padded_img

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

@jit(nopython=True, fastmath=True)
def jit_full_pipeline(padded_img, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level, conversion_mtx, gamma_lut):
    """
    Processes a padded Bayer image to an RGB image using a fused, JIT-compiled pipeline.
    """
    H, W = padded_img.shape
    h, w = H - 2, W - 2
    final_img = np.empty((h, w, 3), dtype=np.float64)
    lut_max_index = len(gamma_lut) - 1

    # Allocate three buffers for prev, curr, next lines for circular buffer (simulated by a list)
    line_buffers = [np.empty(W, dtype=np.float64) for _ in range(3)]

    # Initial fill of the first two line buffers
    for c in range(W):
        line_buffers[0][c] = jit_white_balance_pixel(padded_img[0, c], 0, c, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level)
    for c in range(W):
        line_buffers[1][c] = jit_white_balance_pixel(padded_img[1, c], 1, c, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level)

    # Main loop processing row by row
    for r in range(1, H - 1):
        wb_line_prev = line_buffers[(r - 1) % 3]
        wb_line_curr = line_buffers[r % 3]
        wb_line_next = line_buffers[(r + 1) % 3]

        # Pre-calculate the next white-balanced line
        for c in range(W):
            wb_line_next[c] = jit_white_balance_pixel(padded_img[r + 1, c], r + 1, c, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level)

        for c in range(1, W - 1):
            # --- Debayer ---
            r_val, g_val, b_val = 0.0, 0.0, 0.0
            
            # Use original image index for pattern
            is_row_even = ((r - 1) % 2 == 0)
            is_col_even = ((c - 1) % 2 == 0)

            if pattern_is_bggr:
                if is_row_even and is_col_even: # Blue
                    b_val = wb_line_curr[c]
                    g_val = (wb_line_curr[c-1] + wb_line_curr[c+1] + wb_line_prev[c] + wb_line_next[c]) / 4.0
                    r_val = (wb_line_prev[c-1] + wb_line_prev[c+1] + wb_line_next[c-1] + wb_line_next[c+1]) / 4.0
                elif is_row_even and not is_col_even: # Green
                    g_val = wb_line_curr[c]
                    b_val = (wb_line_curr[c-1] + wb_line_curr[c+1]) / 2.0
                    r_val = (wb_line_prev[c] + wb_line_next[c]) / 2.0
                elif not is_row_even and is_col_even: # Green
                    g_val = wb_line_curr[c]
                    r_val = (wb_line_curr[c-1] + wb_line_curr[c+1]) / 2.0
                    b_val = (wb_line_prev[c] + wb_line_next[c]) / 2.0
                else: # Red
                    r_val = wb_line_curr[c]
                    g_val = (wb_line_curr[c-1] + wb_line_curr[c+1] + wb_line_prev[c] + wb_line_next[c]) / 4.0
                    b_val = (wb_line_prev[c-1] + wb_line_prev[c+1] + wb_line_next[c-1] + wb_line_next[c+1]) / 4.0
            else: # RGGB
                if is_row_even and is_col_even: # Red
                    r_val = wb_line_curr[c]
                    g_val = (wb_line_curr[c-1] + wb_line_curr[c+1] + wb_line_prev[c] + wb_line_next[c]) / 4.0
                    b_val = (wb_line_prev[c-1] + wb_line_prev[c+1] + wb_line_next[c-1] + wb_line_next[c+1]) / 4.0
                elif is_row_even and not is_col_even: # Green
                    g_val = wb_line_curr[c]
                    r_val = (wb_line_curr[c-1] + wb_line_curr[c+1]) / 2.0
                    b_val = (wb_line_prev[c] + wb_line_next[c]) / 2.0
                elif not is_row_even and is_col_even: # Green
                    g_val = wb_line_curr[c]
                    b_val = (wb_line_curr[c-1] + wb_line_curr[c+1]) / 2.0
                    r_val = (wb_line_prev[c] + wb_line_next[c]) / 2.0
                else: # Blue
                    b_val = wb_line_curr[c]
                    g_val = (wb_line_curr[c-1] + wb_line_curr[c+1] + wb_line_prev[c] + wb_line_next[c]) / 4.0
                    r_val = (wb_line_prev[c-1] + wb_line_prev[c+1] + wb_line_next[c-1] + wb_line_next[c+1]) / 4.0
            
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

            final_img[r-1, c-1, 0] = gamma_lut[r_idx]
            final_img[r-1, c-1, 1] = gamma_lut[g_idx]
            final_img[r-1, c-1, 2] = gamma_lut[b_idx]

    return final_img

# @jit(nopython=True, fastmath=True)
def raw_processing_jit_V2(img: np.ndarray, 
                         black_level: int,
                         ADC_max_level: int,
                         bayer_pattern: str, 
                         wb_params: tuple, 
                         fwd_mtx: np.ndarray, 
                         render_mtx: np.ndarray,
                         gamma: str = 'BT709',
                         ) -> np.ndarray:
    """
    Processes a Bayer RAW image to an RGB image using the fully fused V2 JIT pipeline.
    """
    # 1. Initial Conversion and Padding (now handled by JIT function)
    padded_img = jit_prepare_padded_image(img, black_level)
    
    # 2. Prepare parameters
    r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC = wb_params
    pattern_is_bggr = (bayer_pattern == 'BGGR')
    clip_max_level = float(ADC_max_level - black_level)
    conversion_mtx = np.dot(render_mtx, fwd_mtx)
    
    # 3. Call the fully fused JIT pipeline
    if gamma == 'BT709':
        final_float = jit_full_pipeline(padded_img, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level, conversion_mtx, BT709_LUT)
    else:
        # Fallback for non-BT709 gamma is not implemented in the fused JIT function
        # You can add it here if needed, or raise an error.
        raise NotImplementedError("Only BT709 gamma is supported in the JIT V2 pipeline.")

    return final_float
