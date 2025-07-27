# Big loop, padding on-the-fly, no parallelization
# 32.289 ± 0.739 ms
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

# Optional: turn off bounds checking and negative indexing for speed
# This is generally safe when you know your indices are within bounds
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport round


# 400.030 ± 3.591 ms

# This function remains outside JIT as it uses NumPy features not fully supported by Numba's AOT compilation
def create_bt709_lut(size=65536): # 这里是否需要使用cdef？如果不用cdef，意思是效率和普通python一样的意思吗？
    """
    Creates a lookup table (LUT) for BT.709 Gamma correction.
    """
    cdef np.ndarray[np.float32_t, ndim=1] linear_input
    cdef np.ndarray[np.float32_t, ndim=1] gamma_corrected_output
    linear_input = np.linspace(0, 1, size, dtype=np.float32)
    gamma_corrected_output = np.where(linear_input < 0.018,
                                      4.5 * linear_input,
                                      1.099 * (linear_input ** 0.45) - 0.099)
    return gamma_corrected_output.astype(np.float32)

BT709_LUT = create_bt709_lut()

cdef float cy_get_padded_pixel_value(np.ndarray[np.uint16_t, ndim=2] img, int black_level, int r_padded, int c_padded, int H_orig, int W_orig):
    """
    Gets a pixel value from the original image, applying black level correction and reflect padding logic.
    r_padded, c_padded are the logical coordinates in the padded image.
    """
    cdef int r_orig = r_padded - 1
    cdef int c_orig = c_padded - 1
    cdef float val # 是否需要显式声明float32，后续可能要SIMD优化？

    # Apply reflect padding logic
    if r_orig < 0:
        r_orig = -r_orig
    elif r_orig >= H_orig:
        r_orig = H_orig - 2 - (r_orig - H_orig)

    if c_orig < 0:
        c_orig = -c_orig
    elif c_orig >= W_orig:
        c_orig = W_orig - 2 - (c_orig - W_orig)

    val = <float>(img[r_orig, c_orig] - black_level)
    if val < 0:
        val = 0
    return val

cdef float cy_white_balance_pixel(float pixel_val, int r, int c, float r_gain, float g_gain, float b_gain, float r_dBLC, float g_dBLC, float b_dBLC, bint pattern_is_bggr, float clip_max_level):
    """
    Performs white balance for a single pixel.
    """
    cdef bint is_row_even = ((r - 1) % 2 == 0) # 为什么是bint而不是bool？
    cdef bint is_col_even = ((c - 1) % 2 == 0)
    
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

cdef np.ndarray[np.float32_t, ndim=3] cy_full_pipeline(np.ndarray[np.uint16_t, ndim=2] img, int black_level, float r_gain, float g_gain, float b_gain, float r_dBLC, float g_dBLC, float b_dBLC, bint pattern_is_bggr, float clip_max_level, np.ndarray[np.float32_t, ndim=2] conversion_mtx, np.ndarray[np.float32_t, ndim=1] gamma_lut):
    """
    Processes a Bayer image to an RGB image using a fully fused, Cython-compiled pipeline with on-the-fly padding.
    """
    cdef int H_orig, W_orig, H_padded, W_padded, h, w
    cdef np.ndarray[np.float32_t, ndim=3] final_img

    cdef int lut_max_index

    cdef np.ndarray[np.float32_t, ndim=2] line_buffers # 3xW_padded buffer for prev, curr, next lines
    cdef int prev_idx, curr_idx, next_idx
    cdef np.float32_t[:] wb_line_prev, wb_line_curr, wb_line_next # Memoryviews for efficient access

    cdef int r_padded, c_padded
    cdef float r_val, g_val, b_val
    cdef bint is_row_even, is_col_even
    cdef float r_norm, g_norm, b_norm
    cdef float r_ccm, g_ccm, b_ccm
    cdef int r_idx, g_idx, b_idx

    H_orig = <int>img.shape[0]
    W_orig = <int>img.shape[1] # Assuming the image wont exceed int32
    H_padded, W_padded = H_orig + 2, W_orig + 2 # H_padded, W_padded refer to the logical padded dimensions
    h, w = H_orig, W_orig # h, w refer to the final output image dimensions
    final_img = np.empty((h, w, 3), dtype=np.float32)
    lut_max_index = <int>len(gamma_lut) - 1

    # Allocate three buffers for prev, curr, next lines for circular buffer
    line_buffers = np.empty((3, W_padded), dtype=np.float32)

    # Initial fill of the first two line buffers (corresponding to padded_img rows 0 and 1)
    # line_buffers[0] will be prev (logical padded row 0)
    # line_buffers[1] will be curr (logical padded row 1)
    for c_padded in range(W_padded):
        line_buffers[0, c_padded] = cy_white_balance_pixel(
            cy_get_padded_pixel_value(img, black_level, 0, c_padded, H_orig, W_orig),
            0, c_padded, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level
        )
    for c_padded in range(W_padded):
        line_buffers[1, c_padded] = cy_white_balance_pixel(
            cy_get_padded_pixel_value(img, black_level, 1, c_padded, H_orig, W_orig),
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

        # Get memoryviews for current line buffers
        wb_line_prev = line_buffers[prev_idx]
        wb_line_curr = line_buffers[curr_idx]
        wb_line_next = line_buffers[next_idx]

        # Pre-calculate the next white-balanced line into wb_line_next buffer
        # This corresponds to logical padded row r_padded + 1
        for c_padded in range(W_padded):
            wb_line_next[c_padded] = cy_white_balance_pixel(
                cy_get_padded_pixel_value(img, black_level, r_padded + 1, c_padded, H_orig, W_orig),
                r_padded + 1, c_padded, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level
            )

        # Process columns for the current output row (r_padded - 1)
        # c_padded corresponds to the logical padded column index
        for c_padded in range(1, W_padded - 1):
            # --- Debayer ---
            r_val = 0.0
            g_val = 0.0
            b_val = 0.0
            
            # Use original image index for pattern
            is_row_even = ((r_padded - 1) % 2 == 0)
            is_col_even = ((c_padded - 1) % 2 == 0)

            if pattern_is_bggr:
                if is_row_even and is_col_even: # Blue
                    b_val = wb_line_curr[c_padded]
                    g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / <float>4.0
                    r_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / <float>4.0
                elif is_row_even and not is_col_even: # Green
                    g_val = wb_line_curr[c_padded]
                    b_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / <float>2.0
                    r_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / <float>2.0
                elif not is_row_even and is_col_even: # Green
                    g_val = wb_line_curr[c_padded]
                    r_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / <float>2.0
                    b_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / <float>2.0
                else: # Red
                    r_val = wb_line_curr[c_padded]
                    g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / <float>4.0
                    b_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / <float>4.0
            else: # RGGB
                if is_row_even and is_col_even: # Red
                    r_val = wb_line_curr[c_padded]
                    g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / <float>4.0
                    b_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / <float>4.0
                elif is_row_even and not is_col_even: # Green
                    g_val = wb_line_curr[c_padded]
                    r_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / <float>2.0
                    b_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / <float>2.0
                elif not is_row_even and is_col_even: # Green
                    g_val = wb_line_curr[c_padded]
                    b_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / <float>2.0
                    r_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / <float>2.0
                else: # Blue
                    b_val = wb_line_curr[c_padded]
                    g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / <float>4.0
                    r_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / <float>4.0
            
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

            r_idx = <int>round(r_ccm * lut_max_index) # 这种<int>的语法是什么意思？前面不是已经声明了是int吗？
            g_idx = <int>round(g_ccm * lut_max_index)
            b_idx = <int>round(b_ccm * lut_max_index)

            final_img[r_padded-1, c_padded-1, 0] = gamma_lut[r_idx]
            final_img[r_padded-1, c_padded-1, 1] = gamma_lut[g_idx]
            final_img[r_padded-1, c_padded-1, 2] = gamma_lut[b_idx]

    return final_img

def raw_processing_cy(img: np.ndarray, 
                         black_level: int,
                         ADC_max_level: int,
                         bayer_pattern: str, 
                         wb_params: tuple, 
                         fwd_mtx: np.ndarray, 
                         render_mtx: np.ndarray,
                         gamma: str = 'BT709',
                         ) -> np.ndarray:
    """
    Processes a Bayer RAW image to an RGB image using the fully fused V3 JIT pipeline with on-the-fly padding.
    """
    # --- Input Validation and Type Conversion ---
    # Ensure input arrays have the correct dtypes to avoid runtime errors with Cython memoryviews.
    # np.asarray is used to avoid unnecessary copies if the dtype is already correct.
    cdef np.ndarray[np.uint16_t, ndim=2] c_img = np.asarray(img, dtype=np.uint16)
    cdef np.ndarray[np.float32_t, ndim=2] c_fwd_mtx = np.asarray(fwd_mtx, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] c_render_mtx = np.asarray(render_mtx, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] c_BT709_LUT = BT709_LUT

    # --- Other variable declarations ---
    cdef float r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC
    cdef bint pattern_is_bggr
    cdef float clip_max_level
    cdef np.ndarray[np.float32_t, ndim=2] conversion_mtx
    cdef np.ndarray[np.float32_t, ndim=3] final_float

    # 1. Prepare parameters
    r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC = wb_params
    pattern_is_bggr = (bayer_pattern == 'BGGR')
    clip_max_level = <float>(ADC_max_level - black_level)
    # The result of np.dot will be float32 since inputs are now guaranteed to be float32
    conversion_mtx = np.dot(c_render_mtx, c_fwd_mtx)

    # 2. Call the fully fused Cython pipeline
    if gamma == 'BT709':
        final_float = cy_full_pipeline(c_img, black_level, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level, conversion_mtx, c_BT709_LUT)
    else:
        # Fallback for non-BT709 gamma is not implemented in the fused JIT function
        # You can add it here if needed, or raise an error.
        raise NotImplementedError("Only BT709 gamma is supported in the Cython V3 pipeline.")

    return final_float
