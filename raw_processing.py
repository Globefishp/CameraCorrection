# raw_processing.py
# 377.604 ± 3.553 ms
# Author: Haiyun Huang & Google Gemini 2.5 Pro
# A useful raw processing lib implemented by numpy.

import numpy as np
import cv2
import warnings
import time # 导入time模块

def raw_processing(img: np.ndarray, 
                   black_level: int,
                   ADC_max_level: int,
                   bayer_pattern: str, 
                   wb_params: tuple, 
                   fwd_mtx: np.ndarray, 
                   render_mtx: np.ndarray,
                   gamma: str = 'BT709',
                   ) -> np.ndarray:
    '''
    Process Bayer RAW image to RGB image.
    Including White balancing, Debayer, CCM, Y Gamma.

    Args:
        img: (np.ndarray), shape (height, width), dtype np.uint16.
        black_level: (int), the black level of the sensor.
        ADC_max_level: (int), the saturation level of the sensor 
            before applying gains. 
        pattern: (str), Bayer pattern, 'RGGB' or 'BGGR'. 
        wb_params: (tuple), (R_gain, G_gain, B_gain, R_dBLC, G_dBLC, B_dBLC), 
            white balancing parameters.
        fwd_mtx: (np.ndarray), shape (3, 3), dtype np.float64. 
            Transform device RGB to XYZ.
        render_mtx: (np.ndarray), shape (3, 3), dtype np.float64.
            Transform XYZ to Target RGB colourspace.
        gamma: (str), gamma curve, 'BT709', 'sRGB' or 'linear'.
            Default is 'BT709'.

    Returns:
        rgb_img: np.ndarray, shape (height, width, 3), dtype np.uint16.
    '''
    # 0. Black correction
    img = img - black_level

    # 1. White balancing
    wb_img = raw_wb(img, *wb_params, pattern=bayer_pattern, 
                    clip_max_level=ADC_max_level-black_level)

    # 2. Debayer
    if bayer_pattern == 'RGGB':
        device_rgb_img = cv2.cvtColor(wb_img, cv2.COLOR_BAYER_RGGB2RGB)
    elif bayer_pattern == 'BGGR':
        device_rgb_img = cv2.cvtColor(wb_img, cv2.COLOR_BAYER_BGGR2RGB)
    
    device_rgb_img = device_rgb_img.astype(np.float64) / 65535
    
    # 3. Transform to target RGB space
    conversion_mtx = np.dot(render_mtx, fwd_mtx)
    rgb_img = np.dot(device_rgb_img, conversion_mtx.T)
    
    # 5. Y Gamma
    if gamma == 'BT709':
        rgb_img = linear_to_BT709_LUT(rgb_img) # 使用LUT优化版本
    elif gamma == 'sRGB':
        rgb_img = linear_to_sRGB(rgb_img)
    
    return rgb_img

def raw_wb(img: np.ndarray,
           r_gain: float, g_gain: float, b_gain: float,
           r_dBLC: float = 0, g_dBLC: float = 0, b_dBLC: float = 0,
           pattern: str = 'BGGR',
           clip_max_level: int = 4063,
           verbose: bool = False
           ) -> np.ndarray:
    '''
    White balancing for Bayer RAW image with vectorized per-channel clipping.

    Args:
        img: np.ndarray, shape (height, width), dtype np.uint16.
        r_gain, g_gain, b_gain: float, gain for each channel.
        r_dBLC, g_dBLC, b_dBLC: float, differential black level compensation for each channel.
        pattern: str, Bayer pattern, 'RGGB' or 'BGGR'.
        clip_max_level: int, the saturation level of the sensor before applying gains.

    Returns:
        wb_img: np.ndarray, shape (height, width), dtype np.uint16.
        White balanced image.
    '''
    # TODO: Optimize the performance.
    if pattern not in ['RGGB', 'BGGR']:
        raise ValueError("Pattern must be 'RGGB' or 'BGGR'")

    # 使用更高精度的float64进行计算
    wb_img = img.astype(np.float64)
    
    # 1. 创建dBLC映射图并扣除
    dblc_map = np.empty_like(wb_img)
    if pattern == 'RGGB':
        dblc_map[0::2, 0::2] = r_dBLC
        dblc_map[0::2, 1::2] = g_dBLC
        dblc_map[1::2, 0::2] = g_dBLC
        dblc_map[1::2, 1::2] = b_dBLC
    elif pattern == 'BGGR':
        dblc_map[0::2, 0::2] = b_dBLC
        dblc_map[0::2, 1::2] = g_dBLC
        dblc_map[1::2, 0::2] = g_dBLC
        dblc_map[1::2, 1::2] = r_dBLC
    
    wb_img -= dblc_map
    np.clip(wb_img, 0, None, out=wb_img) # 确保dBLC扣除后没有负值

    # 2. 创建剪裁和归一化映射图
    # 核心逻辑: clip((pixel - dBLC) * gain) 等价于 gain * clip(pixel - dBLC, max_val/gain)
    # 这里的实现方式是先裁剪输入，再归一化，效果相同
    max_map = np.empty_like(wb_img)
    # 处理流程：
    # 扣除dBLC
    # 设gain'=1/gain
    # 利用gain'缩放三通道的max_level.
    # Clip信号。（此时，损耗了gain最弱通道的一些Dynamic range）
    # （这一取舍是值得的，因为如果不Clip，则需要猜其他通道的实际值（其他通道已过曝））
    # 利用三通道的max_level，归一化信号。（此时完成wb）
    # Green channels (common for both patterns)
    max_map[0::2, 1::2] = clip_max_level / g_gain
    max_map[1::2, 0::2] = clip_max_level / g_gain

    if pattern == 'RGGB':
        # Red channel
        max_map[0::2, 0::2] = clip_max_level / r_gain
        # Blue channel
        max_map[1::2, 1::2] = clip_max_level / b_gain
        if verbose:
            print(f'raw_wb:RGB clipping level: {max_map[0,0]:.2f}, {max_map[0,1]:.2f}, {max_map[1,1]:.2f}')
    else: # BGGR
        # Blue channel
        max_map[0::2, 0::2] = clip_max_level / b_gain
        # Red channel
        max_map[1::2, 1::2] = clip_max_level / r_gain
        if verbose:
            print(f'raw_wb:RGB clipping level: {max_map[1,1]:.2f}, {max_map[0,1]:.2f}, {max_map[0,0]:.2f}')


    # 3. 使用映射图进行向量化剪裁
    np.clip(wb_img, 0, max_map, out=wb_img)

    # 4. 归一化信号
    # 避免除以零
    max_map[max_map == 0] = 1 
    wb_img = wb_img / max_map * 65535

    # 转换回uint16类型
    return wb_img.astype(np.uint16)

def raw_wb_v2(img: np.ndarray,
           r_gain: float, g_gain: float, b_gain: float,
           r_dBLC: float = 0, g_dBLC: float = 0, b_dBLC: float = 0,
           pattern: str = 'BGGR',
           clip_max_level: int = 4063,
           verbose: bool = False
           ) -> np.ndarray:
    wb_img = img.astype(np.float64)
    # 1. 创建dBLC映射图并扣除
    dblc_map = np.empty_like(wb_img)
    if pattern == 'RGGB':
        dblc_map[0::2, 0::2] = r_dBLC
        dblc_map[0::2, 1::2] = g_dBLC
        dblc_map[1::2, 0::2] = g_dBLC
        dblc_map[1::2, 1::2] = b_dBLC
    elif pattern == 'BGGR':
        dblc_map[0::2, 0::2] = b_dBLC
        dblc_map[0::2, 1::2] = g_dBLC
        dblc_map[1::2, 0::2] = g_dBLC
        dblc_map[1::2, 1::2] = r_dBLC
    
    wb_img -= dblc_map

    # 创建gain映射图
    gain_map = np.empty_like(wb_img)
    if pattern == 'RGGB':
        gain_map[0::2, 0::2] = r_gain
        gain_map[0::2, 1::2] = g_gain
        gain_map[1::2, 0::2] = g_gain
        gain_map[1::2, 1::2] = b_gain
    elif pattern == 'BGGR':
        gain_map[0::2, 0::2] = b_gain
        gain_map[0::2, 1::2] = g_gain
        gain_map[1::2, 0::2] = g_gain
        gain_map[1::2, 1::2] = r_gain
    
    wb_img *= gain_map
    np.clip(wb_img, 0, clip_max_level, out=wb_img) # 确保dBLC扣除后没有负值

    # 4. 归一化信号
    wb_img = wb_img / clip_max_level * 65535

    # 转换回uint16类型
    return wb_img.astype(np.uint16)


def raw_awb(img: np.ndarray,
            roi: tuple[slice, slice],
            roi_2: tuple[slice, slice] = None,
            pattern: str = 'RGGB',
            **kwargs,
            ) -> tuple[np.ndarray, tuple]:
    '''
    Auto White Balance for Bayer RAW image.
    Performs 1-point or 2-point calibration.

    Args:
        img: np.ndarray, shape (height, width), dtype np.uint16.
        roi: tuple[slice(y), slice(x)], region of interest for the first point.
        roi_2: tuple[slice(y), slice(x)], optional region for the second point.
        pattern: str, Bayer pattern, 'RGGB' or 'BGGR'.
        **kwargs: Keyword arguments to be passed to raw_wb, e.g., clip_max_level.

    Returns:
        A tuple containing:
        - wb_img: np.ndarray, the white-balanced image.
        - params: tuple, the calculated (r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC).
    '''
    # 由于共用BLC可能不准确，每通道BLC仍有未扣除部分。
    # 在有色卡的场景下，可以执行两点校准，补偿相对于基准通道(gain最小)之外的其他两个通道的BLC。
    # 典型的场景是：中灰校准后，白色有偏色(如R>G>B)，黑色偏色与白色反向(R<G<B)
    if pattern not in ['RGGB', 'BGGR']:
        raise ValueError("Pattern must be 'RGGB' or 'BGGR'")

    def get_rgb_avg(current_roi):
        roi_img = img[current_roi]
        y_start = current_roi[0].start if current_roi[0].start is not None else 0
        x_start = current_roi[1].start if current_roi[1].start is not None else 0
        y_offset, x_offset = y_start % 2, x_start % 2

        if pattern == 'RGGB':
            r_pix = roi_img[y_offset::2, x_offset::2]
            b_pix = roi_img[1 - y_offset::2, 1 - x_offset::2]
        else: # BGGR
            b_pix = roi_img[y_offset::2, x_offset::2]
            r_pix = roi_img[1 - y_offset::2, 1 - x_offset::2]
        
        g_pix1 = roi_img[y_offset::2, 1 - x_offset::2]
        g_pix2 = roi_img[1 - y_offset::2, x_offset::2]
        
        return (np.mean(r_pix), 
                np.mean(np.concatenate((g_pix1.flatten(), g_pix2.flatten()))), 
                np.mean(b_pix))

    # --- 1-Point AWB (Legacy) ---
    if roi_2 is None:
        avg_r, avg_g, avg_b = get_rgb_avg(roi)
        max_avg = max(avg_r, avg_g, avg_b)
        if max_avg == 0: # Avoid division by zero
            r_gain, g_gain, b_gain = 1.0, 1.0, 1.0
        else:
            g_gain = max_avg / avg_g
            r_gain = max_avg / avg_r
            b_gain = max_avg / avg_b
        params = (r_gain, g_gain, b_gain, 0, 0, 0)
        print(f'raw_awb (1-Point): RGB gain: {r_gain:.2f}, {g_gain:.2f}, {b_gain:.2f}')
    
    # --- 2-Point AWB ---
    else:
        r1, g1, b1 = get_rgb_avg(roi)
        r2, g2, b2 = get_rgb_avg(roi_2)

        # Determine high and low brightness points
        if (r1 + g1 + b1) > (r2 + g2 + b2):
            Rh, Gh, Bh = r1, g1, b1
            Rl, Gl, Bl = r2, g2, b2
        else:
            Rh, Gh, Bh = r2, g2, b2
            Rl, Gl, Bl = r1, g1, b1

        # Calculate slopes
        slope_r = Rh - Rl
        slope_g = Gh - Gl
        slope_b = Bh - Bl
        
        slopes = {'r': slope_r, 'g': slope_g, 'b': slope_b}
        # Avoid division by zero if slopes are identical
        for ch in slopes:
            if slopes[ch] <= 0: slopes[ch] = 1e-6

        # Find reference channel (max slope)
        ref_ch = max(slopes, key=slopes.get)
        
        # Calculate gains and dBLCs
        gains = {}
        dblcs = {}
        
        gains[ref_ch] = 1.0
        dblcs[ref_ch] = 0.0
        
        val_h_ref = {'r': Rh, 'g': Gh, 'b': Bh}[ref_ch]

        for ch in ['r', 'g', 'b']:
            if ch != ref_ch:
                gains[ch] = slopes[ref_ch] / slopes[ch]
                val_h_other = {'r': Rh, 'g': Gh, 'b': Bh}[ch]
                dblcs[ch] = val_h_other - (val_h_ref / gains[ch])

        params = (gains['r'], gains['g'], gains['b'], dblcs['r'], dblcs['g'], dblcs['b'])
        print(f'raw_awb (2-Point): Ref Ch: {ref_ch.upper()}')
        print(f'  Slopes: R={slope_r:.2f}, G={slope_g:.2f}, B={slope_b:.2f}')
        print(f'  Gains: R={params[0]:.4f}, G={params[1]:.4f}, B={params[2]:.4f}')
        print(f'  dBLCs: R={params[3]:.4f}, G={params[4]:.4f}, B={params[5]:.4f}')

    # Call raw_wb with calculated parameters
    wb_img = raw_wb(img, *params, pattern=pattern, **kwargs)
    
    return wb_img, params

def forward_mtx(img, fwd_mtx):
    """
    Transform from device RGB to XYZ in given illuminant.
    """
    xyz_img = np.dot(img, fwd_mtx.T)
    return xyz_img

def XYZ_to_sRGB(xyz_img):
    """
    Transform from XYZ to sRGB.
    """
    srgb_xyz_mtx = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ])
    srgb_img = np.dot(xyz_img, srgb_xyz_mtx.T)
    return srgb_img

def linear_to_sRGB(img):
    """
    Transform from linear to sRGB.
    """
    srgb_img = np.where(img <= 0.0031308,
                        12.92 * img,
                        1.055 * (img ** (1 / 2.4)) - 0.055)
    return srgb_img

def linear_to_BT709(img):
    """
    Transform from linear to BT709.
    """
    bt709_img = np.where(img < 0.018,
                        4.5 * img,
                        1.099 * (img ** 0.45) - 0.099)
    return bt709_img

def create_bt709_lut(size=65536):
    """
    创建BT.709 Gamma校正的查找表 (LUT)。
    """
    # 创建一个从0到1的线性输入值数组
    linear_input = np.linspace(0, 1, size)
    
    # 应用BT.709 Gamma校正公式
    gamma_corrected_output = np.where(linear_input < 0.018,
                                      4.5 * linear_input,
                                      1.099 * (linear_input ** 0.45) - 0.099)
    
    return gamma_corrected_output

# 在模块加载时创建一次LUT
BT709_LUT = create_bt709_lut()

def linear_to_BT709_LUT(img):
    """
    使用查找表 (LUT) 将线性图像转换为BT.709。
    """
    # 裁剪图像值到[0, 1]范围，以防万一
    img_clipped = np.clip(img, 0.0, 1.0)
    
    # 将浮点图像值映射到LUT索引
    # (LUT大小 - 1) 确保索引在 [0, 65535] 范围内
    indices = (img_clipped * (len(BT709_LUT) - 1)).round().astype(np.uint16)
    
    # 使用索引从LUT中获取值
    return BT709_LUT[indices]
