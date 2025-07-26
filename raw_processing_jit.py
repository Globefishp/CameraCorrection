# 82.636 ± 1.808 ms

import numpy as np
from numba import jit
import cv2

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
def jit_white_balance(img_float, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level):
    """
    Performs white balance on a float64 Bayer image.

    Args:
        img_float (np.ndarray): Input float64 Bayer image, black level corrected.
        r_gain, g_gain, b_gain (float): Gains for each channel.
        r_dBLC, g_dBLC, b_dBLC (float): Differential black level compensation.
        pattern_is_bggr (bool): True if Bayer pattern is BGGR, False for RGGB.
        clip_max_level (float): The saturation level.

    Returns:
        np.ndarray: Float64 white-balanced Bayer image.
    """
    height, width = img_float.shape
    wb_img_float = np.empty((height, width), dtype=np.float64)

    for r in range(height):
        for c in range(width):
            pixel_val = img_float[r, c]
            
            is_row_even = (r % 2 == 0)
            is_col_even = (c % 2 == 0)
            
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
            
            wb_img_float[r, c] = pixel_val

    return wb_img_float

@jit(nopython=True, fastmath=True)
def jit_debayer(bayer_img, pattern_is_bggr):
    """
    Performs debayering on a float64 Bayer image using bilinear interpolation.
    Handles boundaries by simple replication of the closest available pixel.

    Args:
        bayer_img (np.ndarray): Input float64 white-balanced Bayer image.
        pattern_is_bggr (bool): True if Bayer pattern is BGGR, False for RGGB.

    Returns:
        np.ndarray: Float64 RGB image with shape (H, W, 3).
    """
    height, width = bayer_img.shape
    rgb_img = np.empty((height, width, 3), dtype=np.float64)

    for r in range(height):
        for c in range(width):
            is_row_even = (r % 2 == 0)
            is_col_even = (c % 2 == 0)

            # Standard RGB channel indices: R=0, G=1, B=2
            ch_r, ch_g, ch_b = 0, 1, 2

            # Bilinear Interpolation
            if pattern_is_bggr:
                if is_row_even:
                    if is_col_even: # Blue pixel
                        rgb_img[r, c, ch_b] = bayer_img[r, c]
                        rgb_img[r, c, ch_g] = (bayer_img[r, max(0, c-1)] + bayer_img[r, min(width-1, c+1)] + bayer_img[max(0, r-1), c] + bayer_img[min(height-1, r+1), c]) / 4.0
                        rgb_img[r, c, ch_r] = (bayer_img[max(0, r-1), max(0, c-1)] + bayer_img[max(0, r-1), min(width-1, c+1)] + bayer_img[min(height-1, r+1), max(0, c-1)] + bayer_img[min(height-1, r+1), min(width-1, c+1)]) / 4.0
                    else: # Green pixel
                        rgb_img[r, c, ch_g] = bayer_img[r, c]
                        rgb_img[r, c, ch_b] = (bayer_img[r, max(0, c-1)] + bayer_img[r, min(width-1, c+1)]) / 2.0
                        rgb_img[r, c, ch_r] = (bayer_img[max(0, r-1), c] + bayer_img[min(height-1, r+1), c]) / 2.0
                else:
                    if is_col_even: # Green pixel
                        rgb_img[r, c, ch_g] = bayer_img[r, c]
                        rgb_img[r, c, ch_r] = (bayer_img[r, max(0, c-1)] + bayer_img[r, min(width-1, c+1)]) / 2.0
                        rgb_img[r, c, ch_b] = (bayer_img[max(0, r-1), c] + bayer_img[min(height-1, r+1), c]) / 2.0
                    else: # Red pixel
                        rgb_img[r, c, ch_r] = bayer_img[r, c]
                        rgb_img[r, c, ch_g] = (bayer_img[r, max(0, c-1)] + bayer_img[r, min(width-1, c+1)] + bayer_img[max(0, r-1), c] + bayer_img[min(height-1, r+1), c]) / 4.0
                        rgb_img[r, c, ch_b] = (bayer_img[max(0, r-1), max(0, c-1)] + bayer_img[max(0, r-1), min(width-1, c+1)] + bayer_img[min(height-1, r+1), max(0, c-1)] + bayer_img[min(height-1, r+1), min(width-1, c+1)]) / 4.0
            else: # RGGB
                if is_row_even:
                    if is_col_even: # Red pixel
                        rgb_img[r, c, ch_r] = bayer_img[r, c]
                        rgb_img[r, c, ch_g] = (bayer_img[r, max(0, c-1)] + bayer_img[r, min(width-1, c+1)] + bayer_img[max(0, r-1), c] + bayer_img[min(height-1, r+1), c]) / 4.0
                        rgb_img[r, c, ch_b] = (bayer_img[max(0, r-1), max(0, c-1)] + bayer_img[max(0, r-1), min(width-1, c+1)] + bayer_img[min(height-1, r+1), max(0, c-1)] + bayer_img[min(height-1, r+1), min(width-1, c+1)]) / 4.0
                    else: # Green pixel
                        rgb_img[r, c, ch_g] = bayer_img[r, c]
                        rgb_img[r, c, ch_r] = (bayer_img[r, max(0, c-1)] + bayer_img[r, min(width-1, c+1)]) / 2.0
                        rgb_img[r, c, ch_b] = (bayer_img[max(0, r-1), c] + bayer_img[min(height-1, r+1), c]) / 2.0
                else:
                    if is_col_even: # Green pixel
                        rgb_img[r, c, ch_g] = bayer_img[r, c]
                        rgb_img[r, c, ch_b] = (bayer_img[r, max(0, c-1)] + bayer_img[r, min(width-1, c+1)]) / 2.0
                        rgb_img[r, c, ch_r] = (bayer_img[max(0, r-1), c] + bayer_img[min(height-1, r+1), c]) / 2.0
                    else: # Blue pixel
                        rgb_img[r, c, ch_b] = bayer_img[r, c]
                        rgb_img[r, c, ch_g] = (bayer_img[r, max(0, c-1)] + bayer_img[r, min(width-1, c+1)] + bayer_img[max(0, r-1), c] + bayer_img[min(height-1, r+1), c]) / 4.0
                        rgb_img[r, c, ch_r] = (bayer_img[max(0, r-1), max(0, c-1)] + bayer_img[max(0, r-1), min(width-1, c+1)] + bayer_img[min(height-1, r+1), max(0, c-1)] + bayer_img[min(height-1, r+1), min(width-1, c+1)]) / 4.0
    return rgb_img

@jit(nopython=True, fastmath=True)
def jit_color_gamma(device_rgb_img, conversion_mtx, gamma_lut):
    """
    Performs color space transformation and gamma correction.

    Args:
        device_rgb_img (np.ndarray): Input float64 RGB image, normalized to [0, 1].
        conversion_mtx (np.ndarray): 3x3 color correction matrix.
        gamma_lut (np.ndarray): 1D lookup table for gamma correction.

    Returns:
        np.ndarray: Final float64 RGB image.
    """
    height, width, _ = device_rgb_img.shape
    final_img = np.empty_like(device_rgb_img, dtype=np.float64)
    lut_max_index = len(gamma_lut) - 1

    for r in range(height):
        for c in range(width):
            r_in, g_in, b_in = device_rgb_img[r, c]
            
            # Correct matrix multiplication logic: [r,g,b] @ M
            r_out = r_in * conversion_mtx[0, 0] + g_in * conversion_mtx[0, 1] + b_in * conversion_mtx[0, 2]
            g_out = r_in * conversion_mtx[1, 0] + g_in * conversion_mtx[1, 1] + b_in * conversion_mtx[1, 2]
            b_out = r_in * conversion_mtx[2, 0] + g_in * conversion_mtx[2, 1] + b_in * conversion_mtx[2, 2]

            if r_out < 0.0: r_out = 0.0
            if r_out > 1.0: r_out = 1.0
            if g_out < 0.0: g_out = 0.0
            if g_out > 1.0: g_out = 1.0
            if b_out < 0.0: b_out = 0.0
            if b_out > 1.0: b_out = 1.0

            r_idx = int(round(r_out * lut_max_index))
            g_idx = int(round(g_out * lut_max_index))
            b_idx = int(round(b_out * lut_max_index))

            final_img[r, c, 0] = gamma_lut[r_idx]
            final_img[r, c, 1] = gamma_lut[g_idx]
            final_img[r, c, 2] = gamma_lut[b_idx]
            
    return final_img

def cv2_debayer(wb_img_float, bayer_pattern, clip_max_level):
    """
    [DEPRECATED] Uses OpenCV for debayering, which requires multiple
    data type conversions. Use jit_debayer for a fully JIT-compiled,
    high-performance pipeline.
    
    Receives float64 input, temporarily converts to uint16 for cv2,
    then converts back to float64 output.
    """
    wb_img_uint16 = np.clip(wb_img_float, 0, clip_max_level).astype(np.uint16)
    
    if bayer_pattern == 'RGGB':
        debayered_uint16 = cv2.cvtColor(wb_img_uint16, cv2.COLOR_BAYER_RGGB2RGB)
    else: # BGGR
        debayered_uint16 = cv2.cvtColor(wb_img_uint16, cv2.COLOR_BAYER_BGGR2RGB)
        
    return debayered_uint16.astype(np.float64) / clip_max_level


def raw_processing_jit(img: np.ndarray, 
                       black_level: int,
                       ADC_max_level: int,
                       bayer_pattern: str, 
                       wb_params: tuple, 
                       fwd_mtx: np.ndarray, 
                       render_mtx: np.ndarray,
                       gamma: str = 'BT709',
                       ) -> np.ndarray:
    """
    Processes a Bayer RAW image to an RGB image using a Numba-JIT accelerated pipeline.
    """
    # 1. Initial Conversion
    img_float = (img - black_level).astype(np.float64)
    
    # 2. JIT White Balance
    r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC = wb_params
    pattern_is_bggr = (bayer_pattern == 'BGGR')
    clip_max_level = float(ADC_max_level - black_level)
    wb_float = jit_white_balance(img_float, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, pattern_is_bggr, clip_max_level)

    # 3. JIT Debayer
    debayer_float = jit_debayer(wb_float, pattern_is_bggr) / clip_max_level
    
    # 4. JIT Color and Gamma
    conversion_mtx = np.dot(render_mtx, fwd_mtx)
    
    if gamma == 'BT709':
        final_float = jit_color_gamma(debayer_float, conversion_mtx, BT709_LUT)
    else:
        # Fallback for other gammas or linear
        final_float = np.dot(debayer_float, conversion_mtx.T)
        if gamma == 'sRGB':
            final_float = np.where(final_float <= 0.0031308, 12.92 * final_float, 1.055 * (final_float ** (1 / 2.4)) - 0.055)

    return final_float
