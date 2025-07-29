# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language=c++

# V6: Global Rearrange + Line Buffer Pipeline
# Implemented Gather and Scatter Debayer Algorithm (Bilinear)
# Core Idea:
# 1. Global Rearrange: Convert the entire Bayer data into a well-organized,
#    padded 4-channel buffer in one go to solve data locality issues.
# 2. Cache-Efficient Pipeline: Subsequent calculations (White Balance, Debayer,
#    CCM, Gamma) are performed on line buffers to maximize CPU cache efficiency.
# 3. Branchless Core: The debayering process uses pure bilinear interpolation
#    without data-dependent branches, which is beneficial for SIMD.

# TODO: 1. Refine code, types, names, and comment key data structures. Add const, unify np.float32_t and float. (Done)
#       2. Validate the entire pipeline.
#       3. Optimize buffers and SIMD based on my knowledge.

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

import cython
import numpy as np
cimport numpy as np
# from cython.parallel cimport prange # OpenMP functions can be used by cimporting openmp
from libc.math cimport powf

# ==============================================================================
# 1. Global Preparation Stage
# ==============================================================================

cdef inline float clip_0_1(float x) noexcept nogil:
    """Clips a float value to the [0.0, 1.0] range."""
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x
    
cdef inline float clip_0_maxlevel(float x, float max_level) noexcept nogil:
    """Clips a float value to the [0.0, max_level] range."""
    if x < 0:
        return 0
    elif x > max_level:
        return max_level
    else:
        return x

cdef inline np.uint16_t _get_pixel_v6(
    np.uint16_t[:, ::1] img, int r_orig, int c_orig, int H_orig, int W_orig) noexcept nogil:
    """
    Safely retrieves pixel values from a Bayer image, using "reflect" padding
    for out-of-bounds coordinates.

    Parameters
    ----------
    img : np.uint16_t[:, ::1]
        The original Bayer image memoryview.
    r_orig : int
        The row coordinate of the desired pixel.
    c_orig : int
        The column coordinate of the desired pixel.
    H_orig : int
        The height of the original image.
    W_orig : int
        The width of the original image.

    Returns
    -------
    np.uint16_t
        The pixel value at the specified coordinates.
    """
    if r_orig < 0:
        r_orig = -r_orig
    elif r_orig >= H_orig:
        r_orig = H_orig - 2 - (r_orig - H_orig)

    if c_orig < 0:
        c_orig = -c_orig
    elif c_orig >= W_orig:
        c_orig = W_orig - 2 - (c_orig - W_orig)

    return img[r_orig, c_orig]

cdef void _rearrange_and_pad(
    np.uint16_t[:, ::1] img,
    np.float32_t[:, :, ::1] raw_4ch,
    int black_level,
    ) noexcept nogil:
    """
    Rearranges a Bayer image into a 4-channel format with 1-pixel padding
    on all sides and subtracts the black level.

    This function converts the 2x2 Bayer pattern (e.g., BGGR) into four
    separate channels. The output is padded to simplify boundary handling in
    subsequent processing steps. The channel order in the output buffer depends
    on the Bayer pattern but the physical memory layout is consistent.
    For BGGR, the layout is [B, G1, G2, R].
    For RGGB, the layout is [R, G1, G2, B].

    This operation can potentially be parallelized.

    Parameters
    ----------
    img : np.uint16_t[:, ::1]
        The original Bayer image (H_orig, W_orig).
    raw_4ch : np.float32_t[:, :, ::1]
        The output 4-channel buffer (H_re, 4, W_re), where
        H_re = H_orig / 2 + 2 and W_re = W_orig / 2 + 2.
    black_level : int
        The black level to be subtracted from pixel values.
    """
    cdef int H_orig = <int>img.shape[0]
    cdef int W_orig = <int>img.shape[1]
    cdef int H_re   = <int>raw_4ch.shape[0]
    cdef int W_re   = <int>raw_4ch.shape[2] # Do only once a frame.
    cdef int r_re, c_re, r_orig, c_orig
    cdef float p00, p01, p10, p11

    for r_re in range(H_re):
        r_orig = 2 * (r_re - 1)
        for c_re in range(W_re):
            c_orig = 2 * (c_re - 1)
            
            p00 = <float>(_get_pixel_v6(img, r_orig,     c_orig,     H_orig, W_orig) - black_level)
            p01 = <float>(_get_pixel_v6(img, r_orig,     c_orig + 1, H_orig, W_orig) - black_level)
            p10 = <float>(_get_pixel_v6(img, r_orig + 1, c_orig,     H_orig, W_orig) - black_level)
            p11 = <float>(_get_pixel_v6(img, r_orig + 1, c_orig + 1, H_orig, W_orig) - black_level)

            raw_4ch[r_re, 0, c_re] = p00 # For BGGR: B; for RGGB: R
            raw_4ch[r_re, 1, c_re] = p01 # For BGGR: G1; for RGGB: G1
            raw_4ch[r_re, 2, c_re] = p10 # For BGGR: G2; for RGGB: G2
            raw_4ch[r_re, 3, c_re] = p11 # For BGGR: R; for RGGB: B

cdef void _white_balance_inplace(
    np.float32_t[:, :, ::1] raw_4ch,
    float r_gain, float g_gain, float b_gain,
    float r_dBLC, float g_dBLC, float b_dBLC,
    float clip_max_level,
    int ch_R, int ch_G1, int ch_G2, int ch_B
    ) noexcept nogil:
    """
    Performs in-place white balance on the full 4-channel buffer.

    This function applies digital gains and digital black level correction (dBLC)
    to the R, G, and B channels. The channel indices for R, G1, G2, and B
    must be provided by the caller to match the Bayer pattern.

    Parameters
    ----------
    raw_4ch : np.float32_t[:, :, ::1]
        The 4-channel buffer to be modified in-place.
    r_gain : float
        Gain for the red channel.
    g_gain : float
        Gain for the green channels (G1 and G2).
    b_gain : float
        Gain for the blue channel.
    r_dBLC : float
        Digital black level correction for the red channel.
    g_dBLC : float
        Digital black level correction for the green channels.
    b_dBLC : float
        Digital black level correction for the blue channel.
    clip_max_level : float
        The maximum pixel value after black level subtraction, used for clipping.
    ch_R : int
        Channel index for the R component.
    ch_G1 : int
        Channel index for the G1 (first green) component.
    ch_G2 : int
        Channel index for the G2 (second green) component.
    ch_B : int
        Channel index for the B (blue) component.
    """
    cdef int H_re = <int>raw_4ch.shape[0]
    cdef int W_re = <int>raw_4ch.shape[2]
    cdef int r_re, c_re
    cdef float val

    for r_re in range(H_re):
        # This loop has no data dependency between columns and could be parallelized.
        for c_re in range(W_re):
            # Blue
            val = (raw_4ch[r_re, ch_B, c_re] - b_dBLC) * b_gain
            raw_4ch[r_re, ch_B, c_re] = clip_0_maxlevel(val, clip_max_level)
            
            # Green 1
            val = (raw_4ch[r_re, ch_G1, c_re] - g_dBLC) * g_gain
            raw_4ch[r_re, ch_G1, c_re] = clip_0_maxlevel(val, clip_max_level)

            # Green 2
            val = (raw_4ch[r_re, ch_G2, c_re] - g_dBLC) * g_gain
            raw_4ch[r_re, ch_G2, c_re] = clip_0_maxlevel(val, clip_max_level)

            # Red
            val = (raw_4ch[r_re, ch_R, c_re] - r_dBLC) * r_gain
            raw_4ch[r_re, ch_R, c_re] = clip_0_maxlevel(val, clip_max_level)

cdef inline void _white_balance_row_inplace(
    float [:, ::1] raw_4ch_row,
    const float r_gain, const float g_gain, const float b_gain, const float r_dBLC, const float g_dBLC, const float b_dBLC,
    const float clip_max_level,
    const bint pattern_is_bggr
    ) noexcept nogil:
    """
    Performs in-place white balance on a single row of 4-channel data.

    This is a standalone, zero-latency step designed for the scatter pipeline.

    Parameters
    ----------
    raw_4ch_row : float[:, ::1]
        The 4-channel row buffer to be modified in-place (4, W_re).
    r_gain : const float
        Gain for the red channel.
    g_gain : const float
        Gain for the green channels.
    b_gain : const float
        Gain for the blue channel.
    r_dBLC : const float
        Digital black level correction for the red channel.
    g_dBLC : const float
        Digital black level correction for the green channels.
    b_dBLC : const float
        Digital black level correction for the blue channel.
    clip_max_level : const float
        The maximum pixel value for clipping.
    pattern_is_bggr : const bint
        Flag indicating if the Bayer pattern is BGGR.
    """
    cdef int W_re = <int>raw_4ch_row.shape[1]
    cdef int c_re
    cdef int ch_B, ch_G1, ch_G2, ch_R
    
    # Define channel indices based on Bayer pattern
    if pattern_is_bggr:
        ch_B = 0; ch_G1 = 1; ch_G2 = 2; ch_R = 3 # NOTE: Systax legal?
    else: # RGGB
        ch_R = 0; ch_G1 = 1; ch_G2 = 2; ch_B = 3

    for c_re in range(W_re):
        # Apply WB and clipping
        raw_4ch_row[ch_B, c_re] = clip_0_maxlevel((raw_4ch_row[ch_B, c_re] - b_dBLC) * b_gain, clip_max_level)
        raw_4ch_row[ch_G1, c_re] = clip_0_maxlevel((raw_4ch_row[ch_G1, c_re] - g_dBLC) * g_gain, clip_max_level)
        raw_4ch_row[ch_G2, c_re] = clip_0_maxlevel((raw_4ch_row[ch_G2, c_re] - g_dBLC) * g_gain, clip_max_level)
        raw_4ch_row[ch_R, c_re] = clip_0_maxlevel((raw_4ch_row[ch_R, c_re] - r_dBLC) * r_gain, clip_max_level)

# ==============================================================================
# 2. Processing Pipeline Core
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 GATHER MODE IMPLEMENTATION
# ------------------------------------------------------------------------------

cdef inline void _debayer_gather_rows(
    float[:, ::1] line_prev_4ch,
    float[:, ::1] line_curr_4ch,
    float[:, ::1] line_next_4ch,
    np.float32_t[:, :, ::1] rgb_line_buffer, # (2, 3, W_out)
    int W_out, bint pattern_is_bggr
    ) noexcept nogil:
    """
    Debayers two rows of RGB data from three rows of 4-channel input using
    a "gather" approach with bilinear interpolation.

    The core logic is hardcoded for the physical BGGR pixel layout. The RGGB
    pattern is handled by the caller by swapping the R and B channels in
    subsequent steps, because the debayered output for RGGB will be BGR.

    Parameters
    ----------
    line_prev_4ch : float[:, ::1]
        The previous row from the 4-channel buffer (4, W_re).
    line_curr_4ch : float[:, ::1]
        The current row from the 4-channel buffer (4, W_re).
    line_next_4ch : float[:, ::1]
        The next row from the 4-channel buffer (4, W_re).
    rgb_line_buffer : np.float32_t[:, :, ::1]
        The output buffer for two rows of RGB data (2, 3, W_out).
    W_out : int
        The width of the final output image.
    pattern_is_bggr : bint
        Flag indicating if the Bayer pattern is BGGR. This currently does not
        affect the logic, as the physical layout is assumed.
    """
    cdef int c_re, c_out
    cdef float R, G, B
    # Define physical channel indices (p00, p01, p10, p11) for BGGR layout
    cdef int ch_p00 = 0, ch_p01 = 1, ch_p10 = 2, ch_p11 = 3

    # Create memoryview slices for direct access to output rows
    cdef float[::1] even_r_line = rgb_line_buffer[0, 0]
    cdef float[::1] even_g_line = rgb_line_buffer[0, 1]
    cdef float[::1] even_b_line = rgb_line_buffer[0, 2]
    cdef float[::1] odd_r_line  = rgb_line_buffer[1, 0]
    cdef float[::1] odd_g_line  = rgb_line_buffer[1, 1]
    cdef float[::1] odd_b_line  = rgb_line_buffer[1, 2]

    # --- Process first output row (even row, from p00/p01 sites) ---
    for c_out in range(W_out):
        c_re = c_out // 2 + 1
        if c_out % 2 == 0: # p00-site (B-site for BGGR)
            B =  line_curr_4ch[ch_p00, c_re  ]
            G = (line_curr_4ch[ch_p01, c_re  ] + line_curr_4ch[ch_p01, c_re-1] + \
                 line_curr_4ch[ch_p10, c_re  ] + line_prev_4ch[ch_p10, c_re  ]) * <float>0.25 
            R = (line_prev_4ch[ch_p11, c_re-1] + line_prev_4ch[ch_p11, c_re  ] + \
                 line_curr_4ch[ch_p11, c_re-1] + line_curr_4ch[ch_p11, c_re  ]) * <float>0.25

        else: # p01-site (G1-site for BGGR)
            G =  line_curr_4ch[ch_p01, c_re  ]
            B = (line_curr_4ch[ch_p00, c_re  ] + line_curr_4ch[ch_p00, c_re+1]) * <float>0.5
            R = (line_prev_4ch[ch_p11, c_re  ] + line_curr_4ch[ch_p11, c_re  ]) * <float>0.5
        even_r_line[c_out] = R
        even_g_line[c_out] = G
        even_b_line[c_out] = B

    # --- Process second output row (odd row, from p10/p11 sites) ---
    for c_out in range(W_out):
        c_re = c_out // 2 + 1
        if c_out % 2 == 0: # p10-site (G2-site for BGGR)
            G =  line_curr_4ch[ch_p10, c_re  ]
            B = (line_curr_4ch[ch_p00, c_re  ] + line_next_4ch[ch_p00, c_re  ]) * <float>0.5
            R = (line_curr_4ch[ch_p11, c_re-1] + line_curr_4ch[ch_p11, c_re  ]) * <float>0.5
        else: # p11-site (R-site for BGGR)
            R =  line_curr_4ch[ch_p11, c_re  ]
            G = (line_curr_4ch[ch_p10, c_re  ] + line_curr_4ch[ch_p10, c_re+1] + \
                 line_curr_4ch[ch_p01, c_re  ] + line_next_4ch[ch_p01, c_re  ]) * <float>0.25
            B = (line_curr_4ch[ch_p00, c_re  ] + line_curr_4ch[ch_p00, c_re+1] + \
                 line_next_4ch[ch_p00, c_re  ] + line_next_4ch[ch_p00, c_re+1]) * <float>0.25

        odd_r_line[c_out] = R
        odd_g_line[c_out] = G
        odd_b_line[c_out] = B

cdef inline void _ccm_gamma_gather_rows(
    np.float32_t[:, :, ::1] rgb_line_buffer, # (2, 3, W_out)
    np.float32_t[:, :, ::1] final_img, 

    int r_out_start, int W_orig,
    const float[3][3] conversion_mtx,
    np.float32_t[::1] gamma_lut,
    float inv_clip_max_level
    ) noexcept nogil:
    """
    Applies Color Correction Matrix (CCM) and gamma correction to a 2-row
    RGB buffer and stores the result in the final image buffer.

    Parameters
    ----------
    rgb_line_buffer : np.float32_t[:, :, ::1]
        Input buffer containing two rows of debayered RGB data (2, 3, W_orig).
    final_img : np.float32_t[:, :, ::1]
        The final output image buffer (H_orig, 3, W_orig).
    r_out_start : int
        The starting row index in `final_img` to write the output to.
    W_orig : int
        The width of the output image.
    conversion_mtx : np.float32_t[:, ::1]
        The 3x3 color conversion matrix.
    gamma_lut : np.float32_t[::1]
        The gamma correction lookup table.
    inv_clip_max_level : float
        The reciprocal of the clipping value, used for normalization.
    """
    cdef int c_out, r_idx, g_idx, b_idx
    cdef float r_val, g_val, b_val, r_ccm, g_ccm, b_ccm
    cdef int lut_max_index = <int>gamma_lut.shape[0] - 1
    cdef float m00 = conversion_mtx[0][0], m01 = conversion_mtx[0][1], m02 = conversion_mtx[0][2]
    cdef float m10 = conversion_mtx[1][0], m11 = conversion_mtx[1][1], m12 = conversion_mtx[1][2]
    cdef float m20 = conversion_mtx[2][0], m21 = conversion_mtx[2][1], m22 = conversion_mtx[2][2]
    cdef int r_offset

    # Process two rows
    # TODO: This section can be rearranged for SIMD friendliness.
    for r_offset in range(2):
        for c_out in range(W_orig):
            r_val = rgb_line_buffer[r_offset, 0, c_out] * inv_clip_max_level
            g_val = rgb_line_buffer[r_offset, 1, c_out] * inv_clip_max_level
            b_val = rgb_line_buffer[r_offset, 2, c_out] * inv_clip_max_level

            r_ccm = r_val * m00 + g_val * m01 + b_val * m02
            g_ccm = r_val * m10 + g_val * m11 + b_val * m12
            b_ccm = r_val * m20 + g_val * m21 + b_val * m22

            r_ccm = clip_0_1(r_ccm)
            g_ccm = clip_0_1(g_ccm)
            b_ccm = clip_0_1(b_ccm)

            r_idx = <int>(r_ccm * lut_max_index + 0.5)
            g_idx = <int>(g_ccm * lut_max_index + 0.5)
            b_idx = <int>(b_ccm * lut_max_index + 0.5)

            final_img[r_out_start + r_offset, 0, c_out] = gamma_lut[r_idx]
            final_img[r_out_start + r_offset, 1, c_out] = gamma_lut[g_idx]
            final_img[r_out_start + r_offset, 2, c_out] = gamma_lut[b_idx]

cdef void _run_pipeline_gather(
    np.float32_t[:, :, ::1] raw_4ch,
    np.float32_t[:, :, ::1] final_img,

    const float[3][3] conversion_mtx,
    np.float32_t[::1] gamma_lut,
    int clip_max_level,
    bint pattern_is_bggr,

    np.float32_t[:, :, ::1] lines_buffer_4ch, # (3, 4, W_orig // 2 + 2)
    np.float32_t[:, :, ::1] rgb_line_buffer, # (2, 3, W_orig)
    ) noexcept nogil:
    """
    Executes the full "gather" mode processing pipeline.

    This function orchestrates the line-buffered processing, including
    debayering, color correction, and gamma correction. It iterates through
    the white-balanced 4-channel data and generates the final RGB image.

    Parameters
    ----------
    raw_4ch : np.float32_t[:, :, ::1]
        The input white-balanced and padded 4-channel Bayer data.
    final_img : np.float32_t[:, :, ::1]
        The output buffer for the final RGB image.
    conversion_mtx : np.float32_t[:, ::1]
        The 3x3 color conversion matrix.
    gamma_lut : np.float32_t[::1]
        The gamma correction lookup table.
    clip_max_level : int
        The maximum ADC value for clipping.
    pattern_is_bggr : bint
        Flag indicating if the Bayer pattern is BGGR.
    lines_buffer_4ch : np.float32_t[:, :, ::1]
        A circular buffer to hold 3 rows of 4-channel data.
    rgb_line_buffer : np.float32_t[:, :, ::1]
        A buffer to hold 2 rows of intermediate RGB data.
    """
    cdef int H_re  = <int>raw_4ch.shape[0]
    cdef int W_out = <int>final_img.shape[2]
    cdef int r_re
    cdef float inv_clip_max_level = <float>1.0 / clip_max_level

    # --- Buffer setup ---
    # Create and initialize line buffer views
    cdef float[:, ::1] line_prev_4ch = lines_buffer_4ch[0]
    cdef float[:, ::1] line_curr_4ch = lines_buffer_4ch[1]
    cdef float[:, ::1] line_next_4ch = lines_buffer_4ch[2]

    # --- Pipeline startup ---
    # Preload the first two rows
    line_prev_4ch[:, :] = raw_4ch[0, :, :]
    line_curr_4ch[:, :] = raw_4ch[1, :, :]

    # --- Main loop ---
    for r_re in range(1, H_re - 1):
        # Load new row
        line_next_4ch[:, :] = raw_4ch[r_re + 1, :, :] # MARK, this systax may not legal in Cython, wait for test.

        # Debayer (Gather)
        _debayer_gather_rows(
            line_prev_4ch, line_curr_4ch, line_next_4ch, 
            rgb_line_buffer, W_out, pattern_is_bggr
            )

        # CCM, Gamma & Store
        _ccm_gamma_gather_rows(
            rgb_line_buffer, final_img, 2 * (r_re - 1), W_out,
            conversion_mtx, gamma_lut, inv_clip_max_level
        )

        # Rotate buffer pointers for the next iteration.
        # This avoids data copying.
        # The rotation cannot be written in one line per Cython limits.
        line_prev_4ch = line_curr_4ch
        line_curr_4ch = line_next_4ch
        line_next_4ch = line_prev_4ch

# ------------------------------------------------------------------------------
# 2.2 SCATTER MODE IMPLEMENTATION
# ------------------------------------------------------------------------------

cdef inline void _debayer_scatter_row(
    const float [:, ::1] wb_4ch_row, # Shape (4, W_re)
    float [:, :, ::1] final_img_padded, # Shape (H_orig + Y_PADDING, 3, W_orig + X_PADDING)
    int r_re,
    int Y_PADDING,
    int X_PADDING   
    ) noexcept nogil:
    """
    Processes a single row of white-balanced 4-channel data, scattering its
    contribution to the padded output buffer.

    Parameters
    ----------
    wb_4ch_row : const float [:, ::1]
        A single row of white-balanced 4-channel data (4, W_re).
    final_img_padded : float [:, :, ::1]
        The padded output buffer where contributions are accumulated.
    r_re : int
        The current row index in the rearranged 4-channel space.
    Y_PADDING : int
        The total vertical padding in the output buffer.
    X_PADDING : int
        The total horizontal padding in the output buffer.
    """
    cdef int W_re = <int>wb_4ch_row.shape[1] # TODO: 这种参数和隐式转换应该被优化掉
    cdef int c_re, y_p00, x_p00
    cdef float B, G1, G2, R
    
    # Define physical channel indices (p00, p01, p10, p11) for (B, G1, G2, R) layout
    cdef int ch_p00 = 0, ch_p01 = 1, ch_p10 = 2, ch_p11 = 3

    # Iterate over all 2x2 pixel blocks in the current row
    for c_re in range(W_re):
        # 1. Read 4 channel values
        B  = wb_4ch_row[ch_p00, c_re]
        G1 = wb_4ch_row[ch_p01, c_re]
        G2 = wb_4ch_row[ch_p10, c_re]
        R  = wb_4ch_row[ch_p11, c_re]

        # 2. Calculate base coordinates in the padded output image
        y_p00 = 2 * (r_re - 1) + Y_PADDING // 2 # Total padding on both side, thus //2
        x_p00 = 2 * (c_re - 1) + X_PADDING // 2 
        # Loop start at y_p00, x_p00 = 1, 1, where is an ineffective pixel
        # Effective area starts from (3, 3) ends at (-4, -4)
        # 3. Scatter contributions (BGGR Pattern)
        # Output channels: 0=R, 1=G, 2=B
        
        # --- R Channel Contributes to... ---
        # R-site (p11)
        final_img_padded[y_p00 + 1, 0, x_p00 + 1] += R # 自己
        # G-sites (p01, p10)
        final_img_padded[y_p00,     0, x_p00 + 1] += R * <float>0.5 # 同block p01
        final_img_padded[y_p00 + 1, 0, x_p00    ] += R * <float>0.5 # 同block p10
        final_img_padded[y_p00 + 1, 0, x_p00 + 2] += R * <float>0.5 # 右侧block p10
        final_img_padded[y_p00 + 2, 0, x_p00 + 1] += R * <float>0.5 # 下方block p01
        # B-site (p00)
        final_img_padded[y_p00,     0, x_p00    ] += R * <float>0.25 # 同block p00
        final_img_padded[y_p00,     0, x_p00 + 2] += R * <float>0.25 # 右侧block p00
        final_img_padded[y_p00 + 2, 0, x_p00    ] += R * <float>0.25 # 下方block p00
        final_img_padded[y_p00 + 2, 0, x_p00 + 2] += R * <float>0.25 # 下方右侧block p00

        # --- G1 Channel Contributes to... ---
        # G-site (p01)
        final_img_padded[y_p00,     1, x_p00 + 1] += G1 # 自己
        # R-site (p11)
        final_img_padded[y_p00 - 1, 1, x_p00 + 1] += G1 * <float>0.25 # 上方block p11
        final_img_padded[y_p00 + 1, 1, x_p00 + 1] += G1 * <float>0.25 # 同block p11
        # B-site (p00)
        final_img_padded[y_p00,     1, x_p00    ] += G1 * <float>0.25 # 同block p00
        final_img_padded[y_p00,     1, x_p00 + 2] += G1 * <float>0.25 # 右侧block p00

        # --- G2 Channel Contributes to... ---
        # G-site (p10)
        final_img_padded[y_p00 + 1, 1, x_p00    ] += G2 # 自己
        # R-site (p11)
        final_img_padded[y_p00 + 1, 1, x_p00 - 1] += G2 * <float>0.25 # 左侧block p11
        final_img_padded[y_p00 + 1, 1, x_p00 + 1] += G2 * <float>0.25 # 同block p11
        # B-site (p00)
        final_img_padded[y_p00,     1, x_p00    ] += G2 * <float>0.25 # 同block p00
        final_img_padded[y_p00 + 2, 1, x_p00    ] += G2 * <float>0.25 # 下方block p00

        # --- B Channel Contributes to... ---
        # B-site (p00)
        final_img_padded[y_p00,     2, x_p00    ] += B # 自己
        # G-sites (p01, p10)
        final_img_padded[y_p00,     2, x_p00 - 1] += B * <float>0.5 # 左侧block p01
        final_img_padded[y_p00,     2, x_p00 + 1] += B * <float>0.5 # 同block p01
        final_img_padded[y_p00 - 1, 2, x_p00    ] += B * <float>0.5 # 上方block p10
        final_img_padded[y_p00 + 1, 2, x_p00    ] += B * <float>0.5 # 同block p10
        # R-site (p11)
        final_img_padded[y_p00 - 1, 2, x_p00 - 1] += B * <float>0.25 # 左侧block p11
        final_img_padded[y_p00 - 1, 2, x_p00 + 1] += B * <float>0.25 # 上方block p11
        final_img_padded[y_p00 + 1, 2, x_p00 - 1] += B * <float>0.25 # 左侧下方block p11
        final_img_padded[y_p00 + 1, 2, x_p00 + 1] += B * <float>0.25 # 同block p11

cdef inline void _process_CCM_gamma_scatter_rows(
    float [:, :, ::1] final_img_padded,
    int y_to_calculate,
    int W_padded,
    const float[3][3] ccm,
    const float[::1] gamma_lut,
    float inv_clip_max_level
    ) noexcept nogil:
    """
    Applies CCM and gamma correction to two completed rows in the padded
    output buffer for the "scatter" pipeline.

    This function is called after a pair of rows in `final_img_padded` has
    received all its contributions from the debayering step.

    Parameters
    ----------
    final_img_padded : float [:, :, ::1]
        The padded output buffer, which is modified in-place.
    y_to_calculate : int
        The starting row index of the two-row block to process.
    W_padded : int
        The padded width of the output buffer.
    ccm : const float[3][3]
        The 3x3 color conversion matrix.
    gamma_lut : const float[::1]
        The gamma correction lookup table.
    inv_clip_max_level : float
        The reciprocal of the clipping value for normalization.
    """
    cdef int c_out, r_idx, g_idx, b_idx
    cdef float r_val, g_val, b_val, r_ccm, g_ccm, b_ccm
    cdef int lut_max_index = <int>gamma_lut.shape[0] - 1
    cdef float m00 = ccm[0][0], m01 = ccm[0][1], m02 = ccm[0][2]
    cdef float m10 = ccm[1][0], m11 = ccm[1][1], m12 = ccm[1][2]
    cdef float m20 = ccm[2][0], m21 = ccm[2][1], m22 = ccm[2][2]
    cdef int r_offset, y_idx

    for r_offset in range(2):
        y_idx = y_to_calculate + r_offset
        for c_out in range(W_padded):
            r_val = final_img_padded[y_idx, 0, c_out]
            g_val = final_img_padded[y_idx, 1, c_out]
            b_val = final_img_padded[y_idx, 2, c_out]

            r_ccm = (r_val * m00 + g_val * m01 + b_val * m02) * inv_clip_max_level
            g_ccm = (r_val * m10 + g_val * m11 + b_val * m12) * inv_clip_max_level
            b_ccm = (r_val * m20 + g_val * m21 + b_val * m22) * inv_clip_max_level

            r_ccm = clip_0_1(r_ccm)
            g_ccm = clip_0_1(g_ccm)
            b_ccm = clip_0_1(b_ccm)

            r_idx = <int>(r_ccm * lut_max_index + 0.5)
            g_idx = <int>(g_ccm * lut_max_index + 0.5)
            b_idx = <int>(b_ccm * lut_max_index + 0.5)

            final_img_padded[y_idx, 0, c_out] = gamma_lut[r_idx]
            final_img_padded[y_idx, 1, c_out] = gamma_lut[g_idx]
            final_img_padded[y_idx, 2, c_out] = gamma_lut[b_idx]

cdef void _run_pipeline_scatter(
    float [:, :, ::1] raw_4ch,
    float [:, :, ::1] final_img_padded,

    const float r_gain, const float g_gain, const float b_gain,
    const float r_dBLC, const float g_dBLC, const float b_dBLC,
    const float[3][3] ccm,
    const np.float32_t [::1] gamma_lut,
    bint pattern_is_bggr,
    const int clip_max_level,
    int Y_PADDING, int X_PADDING
    ) noexcept nogil:
    """
    Executes the full "scatter" mode processing pipeline.

    This function orchestrates the entire scatter-based process:
    1. Performs in-place white balance on each incoming row of 4-channel data.
    2. Scatters the debayering contribution of that row to the output buffer.
    3. Pops completed rows, applies CCM and gamma correction.

    Parameters
    ----------
    raw_4ch : float [:, :, ::1]
        The input (un-white-balanced) 4-channel Bayer data.
    final_img_padded : float [:, :, ::1]
        The padded output buffer for the final RGB image.
    r_gain : const float
        Red channel gain for white balance.
    g_gain : const float
        Green channel gain for white balance.
    b_gain : const float
        Blue channel gain for white balance.
    r_dBLC : const float
        Red channel digital black level correction.
    g_dBLC : const float
        Green channel digital black level correction.
    b_dBLC : const float
        Blue channel digital black level correction.
    ccm : const float[3][3]
        The 3x3 color conversion matrix.
    gamma_lut : const np.float32_t [::1]
        The gamma correction lookup table.
    pattern_is_bggr : bint
        Flag indicating if the Bayer pattern is BGGR.
    clip_max_level : const int
        The maximum ADC value for clipping.
    Y_PADDING : int
        Total vertical padding in the output buffer.
    X_PADDING : int
        Total horizontal padding in the output buffer.
    """
    cdef int H_re     = <int>raw_4ch.shape[0]
    cdef int W_padded = <int>final_img_padded.shape[2]
    cdef int r_re, y_to_calculate
    cdef float inv_clip_max_level = <float>1.0 / clip_max_level

    for r_re in range(H_re):
        # Step A: In-place row white balance
        _white_balance_row_inplace(raw_4ch[r_re], 
            r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, <float>clip_max_level, pattern_is_bggr)

        # Step B: Scatter Debayer
        _debayer_scatter_row(raw_4ch[r_re], final_img_padded, r_re, Y_PADDING, X_PADDING)

        # Step C: Pop & Process
        # Due to data dependencies, after processing r_re, the two rows
        # starting at 2*r_re-3 are ready for post-processing.
        y_to_calculate = (2 * r_re - 3) + Y_PADDING // 2
        _process_CCM_gamma_scatter_rows(
            final_img_padded, y_to_calculate, W_padded,
            ccm, gamma_lut, inv_clip_max_level
        )

# ==============================================================================
# 3. Top-Level Functions
# ==============================================================================

cdef np.ndarray[np.float32_t, ndim=1] c_create_bt709_lut(int size=65536) noexcept:
    """
    Creates a lookup table for BT.709 gamma correction.

    Parameters
    ----------
    size : int, optional
        The number of entries in the lookup table, by default 65536.

    Returns
    -------
    np.ndarray[np.float32_t, ndim=1]
        The generated BT.709 LUT.
    """
    cdef np.ndarray[np.float32_t, ndim=1] lut = np.empty(size, dtype=np.float32)
    cdef int i
    cdef float linear_input_f
    for i in range(size):
        linear_input_f = <float>i / <float>(size - 1)
        if linear_input_f < 0.018:
            lut[i] = <float>4.5 * linear_input_f

        else:
            lut[i] = <float>1.099 * powf(linear_input_f, 0.45) - <float>0.099
    return lut

BT709_LUT = c_create_bt709_lut()

cdef void cy_full_pipeline_v6_scatter(
    np.uint16_t[:, ::1] img, 
    np.float32_t[:, :, ::1] final_img_padded,

    int black_level,
    float r_gain, float g_gain, float b_gain, float r_dBLC, float g_dBLC, float b_dBLC,
    bint pattern_is_bggr, int clip_max_level,
    const np.float32_t[:, ::1] conversion_mtx, const np.float32_t [::1] gamma_lut,
    int Y_PADDING, int X_PADDING,

    np.float32_t[:, :, ::1] raw_4ch # (H_orig//2+2, 4, W_orig//2+2)
    ) noexcept nogil:
    """
    Top-level nogil function for the V6 scatter-mode pipeline.

    Orchestrates the full processing sequence for the scatter mode:
    1. Rearranges the input Bayer image into a padded 4-channel buffer.
    2. Runs the scatter pipeline which includes row-wise white balance,
       debayering, CCM, and gamma correction.

    Parameters
    ----------
    img : np.uint16_t[:, ::1]
        Input raw Bayer image.
    final_img_padded : np.float32_t[:, :, ::1]
        Output buffer for the padded final RGB image.
    black_level : int
        Black level of the sensor.
    r_gain, g_gain, b_gain : float
        White balance gains.
    r_dBLC, g_dBLC, b_dBLC : float
        White balance digital black level corrections.
    pattern_is_bggr : bint
        Flag for BGGR Bayer pattern.
    clip_max_level : int
        Maximum ADC value after black level subtraction.
    conversion_mtx : const np.float32_t[:, ::1]
        The 3x3 color conversion matrix.
    gamma_lut : const np.float32_t [::1]
        The gamma correction lookup table.
    Y_PADDING : int
        Total vertical padding.
    X_PADDING : int
        Total horizontal padding.
    raw_4ch : np.float32_t[:, :, ::1]
        Workspace buffer for the 4-channel rearranged data.
    """
    cdef float[3][3] ccm_data # Create a C-style array on the stack
    cdef int i, j

    # Copy the conversion matrix to the stack. For RGGB, the debayer produces
    # BGR data, so we swap the R and B rows of the CCM to compensate.
    if pattern_is_bggr:
        for i in range(3):
            for j in range(3):
                ccm_data[i][j] = conversion_mtx[i, j]
    else: # for RGGB
        for j in range(3):
            ccm_data[0][j] = conversion_mtx[2, j] # New R channel uses old B processing
            ccm_data[1][j] = conversion_mtx[1, j] # G is unchanged
            ccm_data[2][j] = conversion_mtx[0, j] # New B channel uses old R processing
    
    # Step 1: Global rearrange and padding (without white balance)
    _rearrange_and_pad(img, raw_4ch, black_level)

    # Step 2: Run the scatter pipeline
    _run_pipeline_scatter(
        raw_4ch, final_img_padded,
        r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, 
        ccm_data, gamma_lut,
        pattern_is_bggr, clip_max_level,
        Y_PADDING, X_PADDING
    )

cdef void cy_full_pipeline_v6_gather(
    np.uint16_t[:, ::1] img, 
    np.float32_t[:, :, ::1] final_img,

    int black_level,
    float r_gain, float g_gain, float b_gain, float r_dBLC, float g_dBLC, float b_dBLC,
    bint pattern_is_bggr, int clip_max_level,
    np.float32_t[:, ::1] conversion_mtx, np.float32_t[::1] gamma_lut,

    np.float32_t[:, :, ::1] raw_4ch,
    np.float32_t[:, :, ::1] lines_buffer_4ch,
    np.float32_t[:, :, ::1] rgb_line_buffer # (2, 3, W_orig)
    ) noexcept nogil:
    """
    Top-level nogil function for the V6 gather-mode pipeline.

    Orchestrates the full processing sequence for the gather mode:
    1. Rearranges the input Bayer image into a padded 4-channel buffer.
    2. Performs in-place white balance on the entire 4-channel buffer.
    3. Runs the gather pipeline which includes debayering, CCM, and gamma
       correction on a line-by-line basis.

    Parameters
    ----------
    img : np.uint16_t[:, ::1]
        Input raw Bayer image.
    final_img : np.float32_t[:, :, ::1]
        Output buffer for the final RGB image.
    black_level : int
        Black level of the sensor.
    r_gain, g_gain, b_gain : float
        White balance gains.
    r_dBLC, g_dBLC, b_dBLC : float
        White balance digital black level corrections.
    pattern_is_bggr : bint
        Flag for BGGR Bayer pattern.
    clip_max_level : int
        Maximum ADC value after black level subtraction.
    conversion_mtx : np.float32_t[:, ::1]
        The 3x3 color conversion matrix.
    gamma_lut : np.float32_t[::1]
        The gamma correction lookup table.
    raw_4ch : np.float32_t[:, :, ::1]
        Workspace buffer for the 4-channel rearranged data.
    lines_buffer_4ch : np.float32_t[:, :, ::1]
        Workspace buffer for 3 rows of 4-channel data.
    rgb_line_buffer : np.float32_t[:, :, ::1]
        Workspace buffer for 2 rows of intermediate RGB data.
    """
    cdef float[3][3] ccm_data # Create a C-style array on the stack

    # For RGGB, the debayer produces BGR data. We compensate by swapping the
    # R and B rows of the CCM. This creates a local copy to avoid side effects.
    if pattern_is_bggr:
        for i in range(3):
            for j in range(3):
                ccm_data[i][j] = conversion_mtx[i, j]
    else: # for RGGB
        for j in range(3):
            ccm_data[0][j] = conversion_mtx[2, j] # New R channel uses old B processing
            ccm_data[1][j] = conversion_mtx[1, j] # G is unchanged
            ccm_data[2][j] = conversion_mtx[0, j] # New B channel uses old R processing

    # Step 1: Global rearrange and padding
    _rearrange_and_pad(img, raw_4ch, black_level)

    # Step 2: In-place white balance on the full buffer
    if pattern_is_bggr:
        _white_balance_inplace(raw_4ch, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, <float>clip_max_level, 
                               ch_R=3, ch_G1=1, ch_G2=2, ch_B=0)
    else: # RGGB
        _white_balance_inplace(raw_4ch, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, <float>clip_max_level, 
                               ch_R=0, ch_G1=1, ch_G2=2, ch_B=3)

    # Step 3: Run the line-buffered processing pipeline
    _run_pipeline_gather(raw_4ch, final_img, ccm_data, gamma_lut, clip_max_level, pattern_is_bggr,
        lines_buffer_4ch, rgb_line_buffer
    )


def raw_processing_cy_V6_1(img: np.ndarray,
                         black_level: int,
                         ADC_max_level: int,
                         bayer_pattern: str,
                         wb_params: tuple,
                         fwd_mtx: np.ndarray,
                         render_mtx: np.ndarray,
                         gamma: str = 'BT709',
                         mode: str = 'gather'
                         ) -> np.ndarray:
    """
    Processes a raw image using the V6 Cython pipeline.

    This function implements a high-performance RAW processing pipeline using
    a "rearrange and pipeline" strategy. It supports two internal processing
    modes: 'gather' and 'scatter', which represent different approaches to
    debayering and pipeline organization.

    Parameters
    ----------
    img : np.ndarray
        The input raw Bayer image as a 2D NumPy array (H, W) of uint16.
        Image dimensions must be even.
    black_level : int
        The sensor's black level.
    ADC_max_level : int
        The maximum possible value from the ADC (e.g., 4095 for 12-bit).
    bayer_pattern : str
        The Bayer pattern, e.g., 'BGGR' or 'RGGB'.
    wb_params : tuple
        A tuple containing white balance parameters:
        (r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC).
    fwd_mtx : np.ndarray
        The forward matrix (e.g., from camera primaries to XYZ).
    render_mtx : np.ndarray
        The rendering matrix (e.g., from XYZ to sRGB).
    gamma : str, optional
        The gamma correction to apply. Currently, only 'BT709' is supported.
        Defaults to 'BT709'.
    mode : str, optional
        The processing pipeline mode, either 'gather' or 'scatter'.
        Defaults to 'gather'.

    Returns
    -------
    np.ndarray
        The final processed sRGB image as a 3D NumPy array (H, W, 3) of float32.

    Raises
    ------
    NotImplementedError
        If a gamma type other than 'BT709' is requested.
    ValueError
        If an invalid mode is specified.
    """
    # --- Parameter & Type Preparation ---
    assert img.shape[0] % 2 == 0 and img.shape[1] % 2 == 0, "Image dimensions must be even"
    cdef np.ndarray[np.uint16_t, ndim=2, mode='c'] c_img = np.ascontiguousarray(img, dtype=np.uint16)
    cdef np.ndarray[np.float32_t, ndim=2] c_fwd_mtx = np.asarray(fwd_mtx, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] c_render_mtx = np.asarray(render_mtx, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] c_BT709_LUT = BT709_LUT

    cdef float r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC
    r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC = wb_params
    cdef bint pattern_is_bggr = (bayer_pattern == 'BGGR')
    cdef int clip_max_level = ADC_max_level - black_level
    cdef np.ndarray[np.float32_t, ndim=2] conversion_mtx = np.dot(c_render_mtx, c_fwd_mtx)

    # --- Memory Allocation ---
    cdef int H_orig = <int>c_img.shape[0]
    cdef int W_orig = <int>c_img.shape[1]
    # Intermediate 4-channel buffer with padding, in (H, C, W) format.
    cdef np.ndarray[np.float32_t, ndim=3] raw_4ch = np.empty((H_orig // 2 + 2, 4, W_orig // 2 + 2), dtype=np.float32)
    # Memory allocation is not allowed in if, so define in advance.
    # Future optimization can wrapped these environment in a class like V4.
    # --- Memory Allocation (Gather) ---
    cdef np.ndarray[np.float32_t, ndim=3] lines_buffer_4ch = np.empty((3, 4, W_orig // 2 + 2), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] rgb_line_buffer = np.empty((2, 3, W_orig), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] final_img = np.empty((H_orig, 3, W_orig), dtype=np.float32)
    # --- Memory Allocation (Scatter) ---
    cdef int Y_PADDING = 6
    cdef int X_PADDING = 6 
    cdef np.ndarray[np.float32_t, ndim=3] final_img_padded = np.zeros((H_orig + Y_PADDING, 3, W_orig + X_PADDING), dtype=np.float32)

    if gamma != 'BT709':
        raise NotImplementedError("Only BT709 gamma is supported in the Cython V6 pipeline.")

    if mode == 'gather':
        # Execute the full pipeline within a nogil context
        cy_full_pipeline_v6_gather(c_img, final_img, 
                                       black_level,
                                       r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC,
                                       pattern_is_bggr, clip_max_level,
                                       conversion_mtx, c_BT709_LUT,
                                       raw_4ch, lines_buffer_4ch, rgb_line_buffer
                                       )
        return np.ascontiguousarray(final_img.transpose(0, 2, 1)) 
    elif mode == 'scatter':
        cy_full_pipeline_v6_scatter(c_img, final_img_padded, 
                                        black_level,
                                        r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC,
                                        pattern_is_bggr, clip_max_level,
                                        conversion_mtx, c_BT709_LUT,
                                        Y_PADDING, X_PADDING,
                                        raw_4ch
                                        )

        # Slice the padding off the final image before returning
        return np.ascontiguousarray(final_img_padded[Y_PADDING // 2: Y_PADDING//2 + H_orig, :, X_PADDING // 2: X_PADDING // 2 + W_orig].transpose(0, 2, 1))

    else:
        raise ValueError("Mode must be 'gather' or 'scatter'")
