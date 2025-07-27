# Big loop, padding on-the-fly, with parallelization
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
from libc.math cimport powf

# This function remains outside JIT as it uses NumPy features not fully supported by Numba's AOT compilation
def create_bt709_lut(size=65536):
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

cdef np.ndarray[np.float32_t, ndim=1] c_create_bt709_lut(int size=65536):
    """
    Creates a lookup table (LUT) for BT.709 Gamma correction in C.
    """
    cdef np.ndarray[np.float32_t, ndim=1] lut = np.empty(size, dtype=np.float32)
    cdef int i
    cdef float linear_input
    for i in range(size):
        linear_input = <float>i / <float>(size - 1)
        if linear_input < 0.018:
            lut[i] = 4.5 * linear_input
        else:
            lut[i] = 1.099 * powf(linear_input, 0.45) - 0.099
    return lut

BT709_LUT = c_create_bt709_lut()

cdef inline float cy_get_padded_pixel_value(
    np.uint16_t[:, ::1] img,
    int black_level, int r_padded, int c_padded, int H_orig, int W_orig) nogil:
    """
    Gets a pixel value from the original image, applying black level correction and reflect padding logic.
    r_padded, c_padded are the logical coordinates in the padded image.
    """
    cdef int r_orig = r_padded - 1
    cdef int c_orig = c_padded - 1
    cdef float val

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

cdef inline float cy_white_balance_pixel(
    float pixel_val, int r, int c,
    float r_gain, float g_gain, float b_gain, float r_dBLC, float g_dBLC, float b_dBLC,
    bint pattern_is_bggr, float clip_max_level) nogil:
    """
    Performs white balance for a single pixel.
    """
    cdef bint is_row_even = ((r - 1) % 2 == 0)
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

cdef void cy_full_pipeline(
    np.uint16_t[:, ::1] img, int black_level,
    float r_gain, float g_gain, float b_gain, float r_dBLC, float g_dBLC, float b_dBLC,
    bint pattern_is_bggr, float clip_max_level,
    np.float32_t[:, ::1] conversion_mtx, np.float32_t[::1] gamma_lut,
    np.float32_t[:, :, ::1] final_img,
    np.float32_t[:, ::1] line_buffers
    ) nogil:
    """
    Processes a Bayer image to an RGB image using a fully fused, Cython-compiled pipeline with on-the-fly padding.
    """
    cdef int H_orig = <int>img.shape[0]
    cdef int W_orig = <int>img.shape[1]
    cdef int H_padded = H_orig + 2
    cdef int W_padded = W_orig + 2
    cdef int lut_max_index = <int>gamma_lut.shape[0] - 1
    cdef int r_padded, c_padded

    # Initial fill of the first two line buffers (corresponding to padded_img rows 0 and 1)
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
    cdef int prev_idx, curr_idx, next_idx
    cdef np.float32_t[:] wb_line_prev, wb_line_curr, wb_line_next
    cdef float r_val, g_val, b_val
    cdef bint is_row_even, is_col_even
    cdef float r_norm, g_norm, b_norm
    cdef float r_ccm, g_ccm, b_ccm
    cdef int r_idx, g_idx, b_idx
    cdef int c_padded_inner
    for r_padded in range(1, H_padded - 1):
        # Thread-local variables
        prev_idx = (r_padded - 1) % 3
        curr_idx = r_padded % 3
        next_idx = (r_padded + 1) % 3
        wb_line_prev = line_buffers[prev_idx]
        wb_line_curr = line_buffers[curr_idx]
        wb_line_next = line_buffers[next_idx]

        # Pre-calculate the next white-balanced line into wb_line_next buffer
        for c_padded_inner in range(W_padded):
            wb_line_next[c_padded_inner] = cy_white_balance_pixel(
                cy_get_padded_pixel_value(img, black_level, r_padded + 1, c_padded_inner, H_orig, W_orig),
                r_padded + 1, c_padded_inner, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level
            )

        # Process columns for the current output row (r_padded - 1)
        for c_padded_inner in range(1, W_padded - 1):
            is_row_even = ((r_padded - 1) % 2 == 0)
            is_col_even = ((c_padded_inner - 1) % 2 == 0)

            if pattern_is_bggr:
                if is_row_even and is_col_even: # Blue
                    b_val = wb_line_curr[c_padded_inner]
                    g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) / <float>4.0
                    r_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) / <float>4.0
                elif is_row_even and not is_col_even: # Green
                    g_val = wb_line_curr[c_padded_inner]
                    b_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) / <float>2.0
                    r_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) / <float>2.0
                elif not is_row_even and is_col_even: # Green
                    g_val = wb_line_curr[c_padded_inner]
                    r_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) / <float>2.0
                    b_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) / <float>2.0
                else: # Red
                    r_val = wb_line_curr[c_padded_inner]
                    g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) / <float>4.0
                    b_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) / <float>4.0
            else: # RGGB
                if is_row_even and is_col_even: # Red
                    r_val = wb_line_curr[c_padded_inner]
                    g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) / <float>4.0
                    b_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) / <float>4.0
                elif is_row_even and not is_col_even: # Green
                    g_val = wb_line_curr[c_padded_inner]
                    r_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) / <float>2.0
                    b_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) / <float>2.0
                elif not is_row_even and is_col_even: # Green
                    g_val = wb_line_curr[c_padded_inner]
                    b_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) / <float>2.0
                    r_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) / <float>2.0
                else: # Blue
                    b_val = wb_line_curr[c_padded_inner]
                    g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) / <float>4.0
                    r_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) / <float>4.0

            r_norm = r_val / clip_max_level
            g_norm = g_val / clip_max_level
            b_norm = b_val / clip_max_level

            r_ccm = r_norm * conversion_mtx[0, 0] + g_norm * conversion_mtx[0, 1] + b_norm * conversion_mtx[0, 2]
            g_ccm = r_norm * conversion_mtx[1, 0] + g_norm * conversion_mtx[1, 1] + b_norm * conversion_mtx[1, 2]
            b_ccm = r_norm * conversion_mtx[2, 0] + g_norm * conversion_mtx[2, 1] + b_norm * conversion_mtx[2, 2]

            if r_ccm < 0.0: r_ccm = 0.0
            elif r_ccm > 1.0: r_ccm = 1.0
            if g_ccm < 0.0: g_ccm = 0.0
            elif g_ccm > 1.0: g_ccm = 1.0
            if b_ccm < 0.0: b_ccm = 0.0
            elif b_ccm > 1.0: b_ccm = 1.0

            r_idx = <int>(r_ccm * lut_max_index + 0.5)
            g_idx = <int>(g_ccm * lut_max_index + 0.5)
            b_idx = <int>(b_ccm * lut_max_index + 0.5)

            final_img[r_padded-1, c_padded_inner-1, 0] = gamma_lut[r_idx]
            final_img[r_padded-1, c_padded_inner-1, 1] = gamma_lut[g_idx]
            final_img[r_padded-1, c_padded_inner-1, 2] = gamma_lut[b_idx]

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
    cdef np.ndarray[np.uint16_t, ndim=2] c_img = np.asarray(img, dtype=np.uint16)
    cdef np.ndarray[np.float32_t, ndim=2] c_fwd_mtx = np.asarray(fwd_mtx, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] c_render_mtx = np.asarray(render_mtx, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] c_BT709_LUT = BT709_LUT

    cdef float r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC
    cdef bint pattern_is_bggr
    cdef float clip_max_level
    cdef np.ndarray[np.float32_t, ndim=2] conversion_mtx

    r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC = wb_params
    pattern_is_bggr = (bayer_pattern == 'BGGR')
    clip_max_level = <float>(ADC_max_level - black_level)
    conversion_mtx = np.dot(c_render_mtx, c_fwd_mtx)

    # Prepare buffers in the GIL-holding part
    cdef int H_orig = <int>c_img.shape[0]
    cdef int W_orig = <int>c_img.shape[1]
    cdef int W_padded = W_orig + 2
    cdef np.ndarray[np.float32_t, ndim=3] final_float = np.empty((H_orig, W_orig, 3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] line_buffers = np.empty((3, W_padded), dtype=np.float32)

    if gamma == 'BT709':
        # The GIL is released by calling the nogil function
        cy_full_pipeline(c_img, black_level, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC,
                         pattern_is_bggr, clip_max_level, conversion_mtx, c_BT709_LUT,
                         final_float, line_buffers)
    else:
        raise NotImplementedError("Only BT709 gamma is supported in the Cython V3 pipeline.")

    return final_float
