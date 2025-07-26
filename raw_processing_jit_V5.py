# Debayer部分代码拆分再内联，实际时间没减少，说明V4版本已经智能内联了。
# 5.621 ± 0.537 ms

import numpy as np
from numba import jit, prange

# LUT创建函数保持不变
def create_bt709_lut(size=65536):
    linear_input = np.linspace(0, 1, size, dtype=np.float32)
    gamma_corrected_output = np.where(linear_input < 0.018,
                                      4.5 * linear_input,
                                      1.099 * (linear_input ** 0.45) - 0.099)
    return gamma_corrected_output.astype(np.float32)

BT709_LUT = create_bt709_lut()

# 辅助函数增加内联
@jit(nopython=True, fastmath=True, inline='always')
def jit_get_padded_pixel_value(img, black_level, r_padded, c_padded, H_orig, W_orig):
    r_orig = r_padded - 1
    c_orig = c_padded - 1
    if r_orig < 0: r_orig = -r_orig
    elif r_orig >= H_orig: r_orig = H_orig - 2 - (r_orig - H_orig)
    if c_orig < 0: c_orig = -c_orig
    elif c_orig >= W_orig: c_orig = W_orig - 2 - (c_orig - W_orig)
    val = img[r_orig, c_orig] - black_level
    return float(val) if val > 0 else 0.0

@jit(nopython=True, fastmath=True, inline='always')
def jit_white_balance_pixel(pixel_val, r, c, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level):
    is_row_even = ((r - 1) % 2 == 0)
    is_col_even = ((c - 1) % 2 == 0)
    if pattern_is_bggr:
        if is_row_even and is_col_even: pixel_val = (pixel_val - b_dBLC) * b_gain
        elif not is_row_even and not is_col_even: pixel_val = (pixel_val - r_dBLC) * r_gain
        else: pixel_val = (pixel_val - g_dBLC) * g_gain
    else: # RGGB
        if is_row_even and is_col_even: pixel_val = (pixel_val - r_dBLC) * r_gain
        elif not is_row_even and not is_col_even: pixel_val = (pixel_val - b_dBLC) * b_gain
        else: pixel_val = (pixel_val - g_dBLC) * g_gain
    if pixel_val < 0: pixel_val = 0
    return pixel_val if pixel_val < clip_max_level else clip_max_level

# V4 新增函数: 并行创建白平衡后的填充图像
@jit(nopython=True, parallel=True, fastmath=True)
def jit_prepare_wb_padded_image(img, black_level, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level):
    """
    Creates a padded and white-balanced intermediate image in parallel.
    """
    H_orig, W_orig = img.shape
    H_padded, W_padded = H_orig + 2, W_orig + 2
    wb_padded_img = np.empty((H_padded, W_padded), dtype=np.float32)

    # 使用 prange 进行并行化
    for r_padded in prange(H_padded):
        for c_padded in range(W_padded):
            # 1. 获取带padding的像素值
            padded_val = jit_get_padded_pixel_value(img, black_level, r_padded, c_padded, H_orig, W_orig)
            # 2. 对其进行白平衡
            wb_padded_img[r_padded, c_padded] = jit_white_balance_pixel(
                padded_val, r_padded, c_padded, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level
            )
            
    return wb_padded_img

# V5 优化，尝试重写分支，减少条件判断，手工内联函数。（没有提升）
@jit(nopython=True, fastmath=True, inline='always')
def _apply_ccm_gamma(r_val, g_val, b_val, final_img, r_out, c_out, 
                     clip_max_level, conversion_mtx, gamma_lut, lut_max_index):
    """
    一个独立的、可维护的函数，负责后续处理。
    由于被内联，它不会产生任何函数调用开销。
    """
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

    final_img[r_out, c_out, 0] = gamma_lut[int(round(r_ccm * lut_max_index))]
    final_img[r_out, c_out, 1] = gamma_lut[int(round(g_ccm * lut_max_index))]
    final_img[r_out, c_out, 2] = gamma_lut[int(round(b_ccm * lut_max_index))]

@jit(nopython=True, fastmath=True, inline='always')
def _process_row_bggr_even(r_out, w, final_img, wb_padded_img, clip_max_level, conversion_mtx, gamma_lut, lut_max_index):
    """处理BGGR模式下的偶数行 (B-G-B-G...)"""
    r_padded = r_out + 1
    wb_line_prev = wb_padded_img[r_padded - 1]
    wb_line_curr = wb_padded_img[r_padded]
    wb_line_next = wb_padded_img[r_padded + 1]

    for c_out in range(w):
        c_padded = c_out + 1
        
        if c_out % 2 == 0: # Blue pixel
            b_val = wb_line_curr[c_padded]
            g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / 4.0
            r_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / 4.0
        else: # Green pixel
            g_val = wb_line_curr[c_padded]
            b_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / 2.0
            r_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / 2.0
            
        _apply_ccm_gamma(r_val, g_val, b_val, final_img, r_out, c_out,
                         clip_max_level, conversion_mtx, gamma_lut, lut_max_index)

@jit(nopython=True, fastmath=True, inline='always')
def _process_row_bggr_odd(r_out, w, final_img, wb_padded_img, clip_max_level, conversion_mtx, gamma_lut, lut_max_index):
    """处理BGGR模式下的奇数行 (G-R-G-R...)"""
    r_padded = r_out + 1
    wb_line_prev = wb_padded_img[r_padded - 1]
    wb_line_curr = wb_padded_img[r_padded]
    wb_line_next = wb_padded_img[r_padded + 1]

    for c_out in range(w):
        c_padded = c_out + 1
        
        if c_out % 2 == 0: # Green pixel
            g_val = wb_line_curr[c_padded]
            r_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / 2.0
            b_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / 2.0
        else: # Red pixel
            r_val = wb_line_curr[c_padded]
            g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / 4.0
            b_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / 4.0
            
        _apply_ccm_gamma(r_val, g_val, b_val, final_img, r_out, c_out,
                         clip_max_level, conversion_mtx, gamma_lut, lut_max_index)

@jit(nopython=True, fastmath=True, inline='always')
def _process_row_rggb_even(r_out, w, final_img, wb_padded_img, clip_max_level, conversion_mtx, gamma_lut, lut_max_index):
    """处理RGGB模式下的偶数行 (R-G-R-G...)"""
    r_padded = r_out + 1
    wb_line_prev = wb_padded_img[r_padded - 1]
    wb_line_curr = wb_padded_img[r_padded]
    wb_line_next = wb_padded_img[r_padded + 1]

    for c_out in range(w):
        c_padded = c_out + 1
        
        if c_out % 2 == 0: # Red pixel
            r_val = wb_line_curr[c_padded]
            g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / 4.0
            b_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / 4.0
        else: # Green pixel
            g_val = wb_line_curr[c_padded]
            r_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / 2.0
            b_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / 2.0
            
        _apply_ccm_gamma(r_val, g_val, b_val, final_img, r_out, c_out,
                         clip_max_level, conversion_mtx, gamma_lut, lut_max_index)

@jit(nopython=True, fastmath=True, inline='always')
def _process_row_rggb_odd(r_out, w, final_img, wb_padded_img, clip_max_level, conversion_mtx, gamma_lut, lut_max_index):
    """处理RGGB模式下的奇数行 (G-B-G-B...)"""
    r_padded = r_out + 1
    wb_line_prev = wb_padded_img[r_padded - 1]
    wb_line_curr = wb_padded_img[r_padded]
    wb_line_next = wb_padded_img[r_padded + 1]

    for c_out in range(w):
        c_padded = c_out + 1
        
        if c_out % 2 == 0: # Green pixel
            g_val = wb_line_curr[c_padded]
            b_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1]) / 2.0
            r_val = (wb_line_prev[c_padded] + wb_line_next[c_padded]) / 2.0
        else: # Blue pixel
            b_val = wb_line_curr[c_padded]
            g_val = (wb_line_curr[c_padded-1] + wb_line_curr[c_padded+1] + wb_line_prev[c_padded] + wb_line_next[c_padded]) / 4.0
            r_val = (wb_line_prev[c_padded-1] + wb_line_prev[c_padded+1] + wb_line_next[c_padded-1] + wb_line_next[c_padded+1]) / 4.0
            
        _apply_ccm_gamma(r_val, g_val, b_val, final_img, r_out, c_out,
                         clip_max_level, conversion_mtx, gamma_lut, lut_max_index)

# V4 修改后的主流水线: 现在可以并行处理
@jit(nopython=True, parallel=True, fastmath=True)
def jit_debayer_CCM_gamma(wb_padded_img, pattern_is_bggr, clip_max_level, conversion_mtx, gamma_lut):
    """
    Processes the intermediate white-balanced image into a final RGB image in parallel.
    """
    H_padded, W_padded = wb_padded_img.shape
    h, w = H_padded - 2, W_padded - 2
    final_img = np.empty((h, w, 3), dtype=np.float32)
    lut_max_index = len(gamma_lut) - 1

    # 使用 prange 并行处理每一行
    # 循环的每次迭代现在是独立的
    for r_out in prange(h):
        if pattern_is_bggr:
            if r_out % 2 == 0: # Even row, BGGR pattern (B-G-B-G...)
                _process_row_bggr_even(r_out, w, final_img, wb_padded_img, clip_max_level, conversion_mtx, gamma_lut, lut_max_index)
            else: # Odd row, BGGR pattern (G-R-G-R...)
                _process_row_bggr_odd(r_out, w, final_img, wb_padded_img, clip_max_level, conversion_mtx, gamma_lut, lut_max_index)
        else: # RGGB pattern
            if r_out % 2 == 0: # Even row, RGGB pattern (R-G-R-G...)
                _process_row_rggb_even(r_out, w, final_img, wb_padded_img, clip_max_level, conversion_mtx, gamma_lut, lut_max_index)
            else: # Odd row, RGGB pattern (G-B-G-B...)
                _process_row_rggb_odd(r_out, w, final_img, wb_padded_img, clip_max_level, conversion_mtx, gamma_lut, lut_max_index)
            
    return final_img

# V4 版本的主调用函数
@jit(nopython=True, fastmath=True) # 这个包装器本身不需要并行
def raw_processing_jit_V5(img: np.ndarray, 
                         black_level: int,
                         ADC_max_level: int,
                         bayer_pattern: str, 
                         wb_params: tuple, 
                         fwd_mtx: np.ndarray, 
                         render_mtx: np.ndarray,
                         gamma: str = 'BT709',
                         ) -> np.ndarray:
    """
    Processes a Bayer RAW image to an RGB image using the parallel-optimized V5 JIT pipeline.
    """
    
    # 1. Prepare parameters
    r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC = wb_params
    pattern_is_bggr = (bayer_pattern == 'BGGR')
    clip_max_level = float(ADC_max_level - black_level)
    conversion_mtx = np.dot(render_mtx, fwd_mtx)
    
    if gamma != 'BT709':
        raise NotImplementedError("Only BT709 gamma is supported in the JIT V4 pipeline.")

    # 2. Jit phase 1: Prepare the white balance padded image (H+2, W+2)
    wb_padded_img = jit_prepare_wb_padded_image(
        img, black_level, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level
    )
    
    # 3. Jit phase 2: Proceeding remaining steps in one jit
    final_float = jit_debayer_CCM_gamma(
        wb_padded_img, pattern_is_bggr, clip_max_level, conversion_mtx, BT709_LUT
    )

    return final_float
