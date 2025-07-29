# Big loop, padding on-the-fly, with parallelization

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language=c++
# V7: Encapsulate pipeline into a C-style class, inspired by V4.

# Optional: turn off bounds checking and negative indexing for speed
# This is generally safe when you know your indices are within bounds

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

# rgb line buffer after line debayer
# Modified from V3.
# SIMD friendly CCM.
# Better padding and WB.

# 18.776 ± 0.462 ms


import cython
import numpy as np
cimport numpy as np
from typing import Tuple
# from libc.math cimport powf
from cython.operator cimport dereference


cdef np.ndarray[np.float32_t, ndim=1] c_create_bt709_lut(int size=65536) noexcept:
    """
    Creates a lookup table (LUT) for BT.709 Gamma correction in C.
    """
    cdef np.ndarray[np.float32_t, ndim=1] lut = np.empty(size, dtype=np.float32)
    cdef int i
    cdef float linear_input_f
    for i in range(size):
        linear_input_f = <float>i / <float>(size - 1)
        if linear_input_f < 0.018:
            lut[i] = <float>4.5 * linear_input_f
        else:
            lut[i] = <float>1.099 * linear_input_f**0.45 - <float>0.099
    return lut


cdef inline float cy_get_padded_pixel_value(
    np.uint16_t[:, ::1] img,
    int black_level, int r_padded, int c_padded, int H_orig, int W_orig) noexcept nogil:
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

cdef void cy_white_balance_line_vectorized(
    np.uint16_t[:, ::1] img, int black_level, int r_padded, int H_orig, int W_orig,
    float r_gain, float g_gain, float b_gain, float r_dBLC, float g_dBLC, float b_dBLC,
    bint pattern_is_bggr, float clip_max_level,
    np.float32_t[::1] out_line_buffer) noexcept nogil:
    """
    Performs white balance for an entire line in a vectorized, SIMD-friendly manner.
    This version separates padding/black-level correction from WB calculation
    to allow for vectorization of the WB step.
    """
    cdef int W_padded = W_orig + 2
    cdef int c_orig, r_ori_idx, c_padded
    cdef int i

    # Step 1: Fill the buffer with black-level corrected values.
    if r_padded == 0:
        # First line, pad using 1 line.
        r_ori_idx = 1
    elif r_padded > H_orig - 1: # Final line, pad using -2 line.
        r_ori_idx = H_orig - 2
    else:
        r_ori_idx = r_padded - 1

    # By creating a 1D view of the row, we simplify the memory access pattern in the loop,
    # which can help the compiler's vectorizer by avoiding complex 2D indexing.
    cdef const np.uint16_t[::1] img_row = img[r_ori_idx]
    for c_orig in range(W_orig):
        out_line_buffer[c_orig + 1] = <float>(img_row[c_orig] - black_level)

    # Handle padded edges manually
    out_line_buffer[0] = <float>(img_row[1] - black_level)
    out_line_buffer[W_padded - 1] = <float>(img_row[W_orig - 3] - black_level)

    # Step 2: Apply white balance in-place using vectorized loops.
    cdef bint is_row_even = ((r_padded - 1) % 2 == 0) # is row even in original bayer image.
    cdef float gain_even, gain_odd, dBLC_even, dBLC_odd, val1, val2

    if pattern_is_bggr:
        if is_row_even: # Row is B G B G...
            gain_even, dBLC_even = b_gain, b_dBLC  # Blue
            gain_odd, dBLC_odd = g_gain, g_dBLC   # Green
        else: # Row is G R G R...
            gain_even, dBLC_even = g_gain, g_dBLC   # Green
            gain_odd, dBLC_odd = r_gain, r_dBLC   # Red
    else: # RGGB
        if is_row_even: # Row is R G R G...
            gain_even, dBLC_even = r_gain, r_dBLC   # Red
            gain_odd, dBLC_odd = g_gain, g_dBLC   # Green
        else: # Row is G B G B...
            gain_even, dBLC_even = g_gain, g_dBLC   # Green
            gain_odd, dBLC_odd = b_gain, b_dBLC   # Blue

    # By merging the two loops, we process adjacent memory locations (i*2 and i*2+1)
    # in the same iteration. This creates a contiguous memory access pattern,
    # which is ideal for SIMD vectorization.
    cdef int c_half = W_padded // 2
    for i in range(c_half):
        # Process even c_padded (corresponds to odd columns in the original pattern of the row)
        c_padded = i * 2
        val1 = (out_line_buffer[c_padded] - dBLC_odd) * gain_odd
        out_line_buffer[c_padded] = max(<float>0.0, min(clip_max_level, val1))

        # Process odd c_padded (corresponds to even columns in the original pattern of the row)
        c_padded = i * 2 + 1
        val2 = (out_line_buffer[c_padded] - dBLC_even) * gain_even
        out_line_buffer[c_padded] = max(<float>0.0, min(clip_max_level, val2))


cdef inline (float, float, float) cy_debayer_pixel(
    np.float32_t[::1] wb_line_prev, np.float32_t[::1] wb_line_curr, np.float32_t[::1] wb_line_next,
    int c_padded_inner, bint is_row_even, bint is_col_even, bint pattern_is_bggr) noexcept nogil:
    cdef float r_val, g_val, b_val

    if pattern_is_bggr:
        if is_row_even and is_col_even: # Blue
            b_val = wb_line_curr[c_padded_inner]
            g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * <float>0.25
            r_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) * <float>0.25
        elif is_row_even and not is_col_even: # Green
            g_val = wb_line_curr[c_padded_inner]
            b_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) * <float>0.5
            r_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * <float>0.5
        elif not is_row_even and is_col_even: # Green
            g_val = wb_line_curr[c_padded_inner]
            r_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) * <float>0.5
            b_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * <float>0.5
        else: # Red
            r_val = wb_line_curr[c_padded_inner]
            g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * <float>0.25
            b_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) * <float>0.25
    else: # RGGB
        if is_row_even and is_col_even: # Red
            r_val = wb_line_curr[c_padded_inner]
            g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * <float>0.25
            b_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) * <float>0.25
        elif is_row_even and not is_col_even: # Green
            g_val = wb_line_curr[c_padded_inner]
            r_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) * <float>0.5
            b_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * <float>0.5
        elif not is_row_even and is_col_even: # Green
            g_val = wb_line_curr[c_padded_inner]
            b_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1]) * <float>0.5
            r_val = (wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * <float>0.5
        else: # Blue
            b_val = wb_line_curr[c_padded_inner]
            g_val = (wb_line_curr[c_padded_inner-1] + wb_line_curr[c_padded_inner+1] + wb_line_prev[c_padded_inner] + wb_line_next[c_padded_inner]) * <float>0.25
            r_val = (wb_line_prev[c_padded_inner-1] + wb_line_prev[c_padded_inner+1] + wb_line_next[c_padded_inner-1] + wb_line_next[c_padded_inner+1]) * <float>0.25
    return r_val, g_val, b_val

cdef void cy_full_pipeline(
    np.uint16_t[:, ::1] img, int black_level,
    float r_gain, float g_gain, float b_gain, float r_dBLC, float g_dBLC, float b_dBLC,
    bint pattern_is_bggr, float clip_max_level,
    np.float32_t[:, ::1] conversion_mtx, np.float32_t[::1] gamma_lut,
    np.float32_t[:, :, ::1] final_img,
    np.float32_t[:, ::1] line_buffers,
    float[:, ::1] rgb_line_buffer,
    float[:, ::1] ccm_line_buffer,
    ) noexcept nogil:
    """
    Processes a Bayer image to an RGB image using a fully fused, Cython-compiled pipeline with on-the-fly padding.
    """
    cdef int H_orig = <int>img.shape[0]
    cdef int W_orig = <int>img.shape[1]
    cdef int H_padded = H_orig + 2
    cdef int W_padded = W_orig + 2
    cdef int lut_max_index = <int>gamma_lut.shape[0] - 1
    cdef float inv_clip_max_level = <float>1.0 / clip_max_level
    cdef int r_padded, c_padded

    cdef float[::1] r_line_buffer = rgb_line_buffer[0]
    cdef float[::1] g_line_buffer = rgb_line_buffer[1]
    cdef float[::1] b_line_buffer = rgb_line_buffer[2]

    cdef float[::1] r_ccm_line = ccm_line_buffer[0]
    cdef float[::1] g_ccm_line = ccm_line_buffer[1]
    cdef float[::1] b_ccm_line = ccm_line_buffer[2]

    cdef float m00 = conversion_mtx[0, 0], m01 = conversion_mtx[0, 1], m02 = conversion_mtx[0, 2]
    cdef float m10 = conversion_mtx[1, 0], m11 = conversion_mtx[1, 1], m12 = conversion_mtx[1, 2]
    cdef float m20 = conversion_mtx[2, 0], m21 = conversion_mtx[2, 1], m22 = conversion_mtx[2, 2]

    # Pre-fill the first two line buffers
    cy_white_balance_line_vectorized(img, black_level, 0, H_orig, W_orig, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level, line_buffers[0])
    cy_white_balance_line_vectorized(img, black_level, 1, H_orig, W_orig, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level, line_buffers[1])

    # Main loop processing row by row (r_padded corresponds to the logical padded row index)
    cdef int prev_idx, curr_idx, next_idx
    cdef np.float32_t[::1] wb_line_prev, wb_line_curr, wb_line_next
    cdef float r_val, g_val, b_val
    cdef float r_norm, g_norm, b_norm
    cdef float r_ccm, g_ccm, b_ccm
    cdef int r_idx, g_idx, b_idx
    cdef bint is_row_even, is_col_even
    cdef int c_padded_inner
    cdef int c_orig_inner

    for r_padded in range(1, H_padded - 1):
        # Thread-local variables
        prev_idx = (r_padded - 1) % 3
        curr_idx = r_padded % 3
        next_idx = (r_padded + 1) % 3
        wb_line_prev = line_buffers[prev_idx]
        wb_line_curr = line_buffers[curr_idx]
        wb_line_next = line_buffers[next_idx]

        # Pre-calculate the next white-balanced line into wb_line_next buffer
        cy_white_balance_line_vectorized(
            img, black_level, r_padded + 1, H_orig, W_orig,
            r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC,
            pattern_is_bggr, clip_max_level, wb_line_next
        )

        # Demosaicing for the current output row, filling the R, G, B line buffers
        for c_padded_inner in range(1, W_padded - 1):
            c_orig_inner = c_padded_inner - 1
            is_row_even = ((r_padded - 1) % 2 == 0)
            is_col_even = ((c_padded_inner - 1) % 2 == 0)

            r_val, g_val, b_val = cy_debayer_pixel(
                wb_line_prev, wb_line_curr, wb_line_next,
                c_padded_inner, is_row_even, is_col_even, pattern_is_bggr
            )
            r_line_buffer[c_orig_inner] = r_val
            g_line_buffer[c_orig_inner] = g_val
            b_line_buffer[c_orig_inner] = b_val

        for c_orig_inner in range(W_orig):
            r_ccm_line[c_orig_inner] = max(0, min(1, (r_line_buffer[c_orig_inner] * inv_clip_max_level) * m00 + \
                                                     (g_line_buffer[c_orig_inner] * inv_clip_max_level) * m01 + \
                                                     (b_line_buffer[c_orig_inner] * inv_clip_max_level) * m02)) * \
                                                      lut_max_index + 0.5
        for c_orig_inner in range(W_orig):
            g_ccm_line[c_orig_inner] = max(0, min(1, (r_line_buffer[c_orig_inner] * inv_clip_max_level) * m10 + \
                                                     (g_line_buffer[c_orig_inner] * inv_clip_max_level) * m11 + \
                                                     (b_line_buffer[c_orig_inner] * inv_clip_max_level) * m12)) * \
                                                     lut_max_index + 0.5
        for c_orig_inner in range(W_orig):
            b_ccm_line[c_orig_inner] = max(0, min(1, (r_line_buffer[c_orig_inner] * inv_clip_max_level) * m20 + \
                                                     (g_line_buffer[c_orig_inner] * inv_clip_max_level) * m21 + \
                                                     (b_line_buffer[c_orig_inner] * inv_clip_max_level) * m22)) * \
                                                     lut_max_index + 0.5

        for c_orig_inner in range(W_orig):
            final_img[r_padded-1, c_orig_inner, 0] = gamma_lut[<int>r_ccm_line[c_orig_inner]]
            final_img[r_padded-1, c_orig_inner, 1] = gamma_lut[<int>g_ccm_line[c_orig_inner]]
            final_img[r_padded-1, c_orig_inner, 2] = gamma_lut[<int>b_ccm_line[c_orig_inner]]


cdef class RawV7Processor:
    # C-level attributes for fast access
    cdef int black_level, H_orig, W_orig
    cdef float clip_max_level
    cdef bint pattern_is_bggr
    cdef float r_gain, g_gain, b_gain
    cdef float r_dBLC, g_dBLC, b_dBLC
    cdef np.ndarray conversion_mtx, gamma_lut
    # Pre-allocated buffers
    cdef np.ndarray line_buffers, rgb_line_buffer, ccm_line_buffer

    def __cinit__(self, int H_orig, int W_orig, int black_level, int ADC_max_level, str bayer_pattern,
                  tuple wb_params, np.ndarray fwd_mtx, np.ndarray render_mtx,
                  str gamma='BT709'):
        """
        Initializes the processor with all constant parameters and pre-allocates buffers.
        """
        # Store parameters
        self.H_orig = H_orig
        self.W_orig = W_orig
        self.black_level = black_level
        self.clip_max_level = <float>(ADC_max_level - black_level)
        self.pattern_is_bggr = (bayer_pattern == 'BGGR')
        
        self.r_gain, self.g_gain, self.b_gain, self.r_dBLC, self.g_dBLC, self.b_dBLC = wb_params

        # Pre-calculate conversion matrix
        cdef np.ndarray[np.float32_t, ndim=2] c_fwd_mtx = np.asarray(fwd_mtx, dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=2] c_render_mtx = np.asarray(render_mtx, dtype=np.float32)
        self.conversion_mtx = np.dot(c_render_mtx, c_fwd_mtx)

        # Generate Gamma LUT
        if gamma == 'BT709':
            self.gamma_lut = c_create_bt709_lut()
        else:
            raise NotImplementedError(f"Gamma '{gamma}' is not supported.")
            
        # Pre-allocate buffers
        cdef int W_padded = self.W_orig + 2
        # cdef np.ndarray[np.float32_t, ndim=2] line_buffers = np.empty((3, W_padded), dtype=np.float32)
        # # Channel-wise RGB line buffer for demosaicing output
        # cdef np.ndarray[np.float32_t, ndim=2] rgb_line_buffer = np.empty((3, W_orig), dtype=np.float32)
        # # Temporary buffer for CCM results to enable SoA processing
        # cdef np.ndarray[np.float32_t, ndim=2] ccm_line_buffer = np.empty((3, W_orig), dtype=np.float32)
        self.line_buffers = np.empty((3, W_padded), dtype=np.float32)
        self.rgb_line_buffer = np.empty((3, W_orig), dtype=np.float32)
        self.ccm_line_buffer = np.empty((3, W_orig), dtype=np.float32)

    def process(self, np.ndarray img):
        """
        Processes a single Bayer RAW image frame using pre-allocated buffers.
        """
        cdef np.ndarray[np.uint16_t, ndim=2] c_img = np.asarray(img, dtype=np.uint16)
        
        # Allocate only the final output buffer for each frame
        cdef np.ndarray[np.float32_t, ndim=3] final_float = np.empty((self.H_orig, self.W_orig, 3), dtype=np.float32)

        # Call the core pipeline with pre-allocated buffers
        cy_full_pipeline(
            c_img, self.black_level,
            self.r_gain, self.g_gain, self.b_gain, self.r_dBLC, self.g_dBLC, self.b_dBLC,
            self.pattern_is_bggr, self.clip_max_level,
            self.conversion_mtx, self.gamma_lut,
            final_float, self.line_buffers,
            self.rgb_line_buffer, self.ccm_line_buffer
        )

        return final_float

def raw_processing_cy_V7(img: np.ndarray,
                         black_level: int,
                         ADC_max_level: int,
                         bayer_pattern: str,
                         wb_params: tuple,
                         fwd_mtx: np.ndarray,
                         render_mtx: np.ndarray,
                         gamma: str = 'BT709',
                         ) -> np.ndarray:
    """
    Python wrapper to instantiate and use the RawV7Processor.
    Note: For processing sequences, the processor should be instantiated only once.
    """
    cdef int H_orig = img.shape[0]
    cdef int W_orig = img.shape[1]
    
    processor = RawV7Processor(H_orig, W_orig, black_level, ADC_max_level, bayer_pattern,
                               wb_params, fwd_mtx, render_mtx, gamma)
    return processor.process(img)
