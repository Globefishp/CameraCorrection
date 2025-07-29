# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language=c++

# V6: 全局重排 + 行缓冲流水线 (Gather Mode)
# 核心思想:
# 1. 一次性全局重排: 将Bayer数据完整转换为一个排布规整、带padding的4通道缓冲，解决数据局部性问题。
# 2. 缓存高效流水线: 后续计算(白平衡、Debayer、CCM、Gamma)在行缓冲上进行，确保CPU缓存效率。
# 3. 无分支计算核心: Debayering采用纯双线性插值，无数据依赖分支，利于SIMD。

# 函数的输入输出主要有两种模式：Python型返回数据，C型传入出参。
# 对于Python性，函数签名为(输入数据流，额外参数)
# 对于C型，函数签名为(输入数据流，(Optional, 在输入输出合并时)输出数据流，额外参数, (Optional)缓冲区)
# TODO: 1. 整理代码，变量类型，名称，关键数据结构注释。加入const, 统一np.float32_t 和float
#       2. 验证全流程。
#       3. 按照我的知识，优化缓冲，优化SIMD。

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
# 1. 全局准备阶段 (Global Preparation Stage)
# ==============================================================================

cdef inline float clip_0_1(float x) noexcept nogil:
    # 具体实现似乎有斟酌的余地？字面量？
    # 如何优化成无分支SSE指令？
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x
    
cdef inline float clip_0_maxlevel(float x, float max_level) noexcept nogil:
    if x < 0:
        return 0
    elif x > max_level:
        return max_level
    else:
        return x

cdef inline np.uint16_t _get_pixel_v6(
    np.uint16_t[:, ::1] img, int r_orig, int c_orig, int H_orig, int W_orig) noexcept nogil:
    """
    安全地获取超过原始Bayer图像边界的像素值，使用reflect方法
    
    params:
    img: 原始Bayer图像，uint16_t类型。大小(H_orig, W_orig)
    r_orig, c_orig: 希望获取的原始图像中的像素坐标。
    H_orig, W_orig: 原始图像的高度和宽度。

    return:
    安全获取的像素值，类型为uint16_t。
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
    ):
    """
    原始Bayer图重排为一个带四边各1像素padding的Bayer4通道图。
    可能可以并行
    此步骤包含黑电平扣除。
    假定输入是BGGR的情况下，输出缓冲区的通道布局为 B=0, G1=1, G2=2, R=3；
    输入是RGGB的情况下，输出的通道布局相应的变为 R=0, G1=1, G2=2, B=3，因为计算逻辑没变，只是内容改变了。
    params:
    img: 原始Bayer图像，uint16_t类型。大小(H_orig, W_orig)
    rearranged_4ch: 出参，的4通道浮点缓冲，float32_t类型。大小(H_orig/2+2, 4, W_orig/2+2) (HCW格式)

    black_level: 黑电平，int类型。
    """
    cdef int H_orig = img.shape[0]
    cdef int W_orig = img.shape[1]
    cdef int H_re = raw_4ch.shape[0]
    cdef int W_re = raw_4ch.shape[2]
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

            raw_4ch[r_re, 0, c_re] = p00 # B
            raw_4ch[r_re, 1, c_re] = p01 # G1
            raw_4ch[r_re, 2, c_re] = p10 # G2
            raw_4ch[r_re, 3, c_re] = p11 # R

cdef void _white_balance_inplace(
    np.float32_t[:, :, ::1] raw_4ch,
    float r_gain, float g_gain, float b_gain,
    float r_dBLC, float g_dBLC, float b_dBLC,
    float clip_max_level,
    int ch_R, int ch_G1, int ch_G2, int ch_B
    ) noexcept nogil:
    """
    在完整的4通道缓冲上原地执行白平衡。通道顺序由调用者定义。
    params:
    raw_4ch: 入参，出参，Bayer4通道浮点图，float32_t类型。大小(H_orig/2+2, 4, W_orig/2+2) (HCW格式)
    r_gain, g_gain, b_gain: 红、绿、蓝通道的增益，float类型。
    r_dBLC, g_dBLC, b_dBLC: 红、绿、蓝通道的dBLC值，float类型。
    clip_max_level: Bayer图像Clip的电平上限，用于归一化，请将int强制转换成float类型再输入。
    ch_R, ch_G1, ch_G2, ch_B: R, G1, G2, B在Bayer4通道图中的通道号索引，int类型。
    """
    cdef int H_re = raw_4ch.shape[0]
    cdef int W_re = raw_4ch.shape[2]
    cdef int r_re, c_re
    cdef float val

    for r_re in range(H_re):
        # 这里RGGB没有数据依赖性，可以并行也许。思考优化。
        for c_re in range(W_re):
            # Blue
            val = (raw_4ch[r_re, ch_B, c_re] - b_dBLC) * b_gain
            val = clip_0_maxlevel(val, clip_max_level)
            raw_4ch[r_re, ch_B, c_re] = val
            
            # Green 1
            val = (raw_4ch[r_re, ch_G1, c_re] - g_dBLC) * g_gain
            val = clip_0_maxlevel(val, clip_max_level)
            raw_4ch[r_re, ch_G1, c_re] = val

            # Green 2
            val = (raw_4ch[r_re, ch_G2, c_re] - g_dBLC) * g_gain
            val = clip_0_maxlevel(val, clip_max_level)
            raw_4ch[r_re, ch_G2, c_re] = val

            # Red
            val = (raw_4ch[r_re, ch_R, c_re] - r_dBLC) * r_gain
            val = clip_0_maxlevel(val, clip_max_level)
            raw_4ch[r_re, ch_R, c_re] = val

cdef inline void _white_balance_row_inplace( # 直接抄之前的inplace函数
    float [:, ::1] raw_4ch_lines,            # 输入兼输出: 未经白平衡的单行4通道数据
    const float r_gain, const float g_gain, const float b_gain, const float r_dBLC, const float g_dBLC, const float b_dBLC,
    const float clip_max_level,
    const bint pattern_is_bggr                   # Bayer pattern
    ):
    """
    在单行4通道缓冲上原地执行白平衡。这是一个独立的、无延迟的步骤。
    params:
    raw_4ch_row: 入参，出参，4通道浮点图，float32_t类型。大小(4, W_re) (CW格式)
    r_gain, g_gain, b_gain: 白平衡增益
    r_dBLC, g_dBLC, b_dBLC: 白平衡 dBLC
    clip_max_level: Bayer图像Clip的电平上限，用于归一化，请将int强制转换成float类型再输入。
    pattern_is_bggr: Bayer pattern 定义。
    """
    cdef int W_re = raw_4ch_lines.shape[1]
    cdef int c_re
    # 定义4通道
    cdef int ch_B, ch_G1, ch_G2, ch_R
    if pattern_is_bggr:
        ch_B = 0
        ch_G1 = 1
        ch_G2 = 2
        ch_R = 3
    else:
        ch_B = 2
        ch_G1 = 1
        ch_G2 = 0
        ch_R = 3

    for c_re in range(W_re):
        raw_4ch_lines[ch_B, c_re] = (raw_4ch_lines[ch_B, c_re] - b_dBLC) * b_gain
        raw_4ch_lines[ch_G1, c_re] = (raw_4ch_lines[ch_G1, c_re] - g_dBLC) * g_gain
        raw_4ch_lines[ch_G2, c_re] = (raw_4ch_lines[ch_G2, c_re] - g_dBLC) * g_gain
        raw_4ch_lines[ch_R, c_re] = (raw_4ch_lines[ch_R, c_re] - r_dBLC) * r_gain
        # 裁剪到[0, clip_max_level]范围
        raw_4ch_lines[ch_B, c_re] = clip_0_maxlevel(raw_4ch_lines[ch_B, c_re], clip_max_level)
        raw_4ch_lines[ch_G1, c_re] = clip_0_maxlevel(raw_4ch_lines[ch_G1, c_re], clip_max_level)
        raw_4ch_lines[ch_G2, c_re] = clip_0_maxlevel(raw_4ch_lines[ch_G2, c_re], clip_max_level)
        raw_4ch_lines[ch_R, c_re] = clip_0_maxlevel(raw_4ch_lines[ch_R, c_re], clip_max_level)

# ==============================================================================
# 2. 计算流水线核心 (Processing Pipeline Core)
# ==============================================================================

# ------------------------------------------------------------------------------
# 2.1 GATHER MODE IMPLEMENTATION
# ------------------------------------------------------------------------------

cdef inline void _debayer_gather_rows(
    float[:, ::1] line_prev_4ch,
    float[:, ::1] line_curr_4ch,
    float[:, ::1] line_next_4ch,
    np.float32_t[:, :, ::1] rgb_line_buffer, # (2, 3, W_out)
    int W_out, bint pattern_is_bggr):
    """
    使用收集(Gather)模式，根据3行4通道输入，计算2行RGB输出。
    核心逻辑硬编码为处理BGGR物理位置->RGB，通过交换输出指针适应RGGB。

    params:
    line_prev_4ch: 入参，前一行4通道缓冲区，float32_t类型。大小(4, W_orig/2+2) (CW格式)
    line_curr_4ch: 入参，当前行4通道缓冲区，float32_t类型。大小(4, W_orig/2+2) (CW格式)
    line_next_4ch: 入参，下一行4通道缓冲区，float32_t类型。大小(4, W_orig/2+2) (CW格式)
    rgb_line_buffer: 出参，2行RGB浮点图，float32_t类型。大小(2, 3, W_out) (HCW格式)
    W_out: 入参，输出图像宽度，等于原始Bayer图像的宽度，int类型。
    pattern_is_bggr: 入参，Bayer模式是否为BGGR，bool类型。决定了四通道的顺序。
    """
    cdef int c_re, c_out
    cdef float R, G, B
    # 定义输入行缓冲的索引
    cdef int prev = 0, curr = 1, next = 2
    # 定义输出行缓冲的索引
    cdef int even = 0, odd = 1
    # 定义4通道缓冲的物理通道索引 (p00, p01, p10, p11)
    cdef int ch_p00 = 0, ch_p01 = 1, ch_p10 = 2, ch_p11 = 3

    cdef float[::1] even_r_line = rgb_line_buffer[0, 0]
    cdef float[::1] even_g_line = rgb_line_buffer[0, 1]
    cdef float[::1] even_b_line = rgb_line_buffer[0, 2]
    cdef float[::1] odd_r_line  = rgb_line_buffer[1, 0]
    cdef float[::1] odd_g_line  = rgb_line_buffer[1, 1]
    cdef float[::1] odd_b_line  = rgb_line_buffer[1, 2]


    # --- 计算第一行输出 (偶数行, p00/p01 row) ---
    for c_out in range(W_out):
        c_re = c_out // 2 + 1
        if c_out % 2 == 0: # p00-site
            B =  line_curr_4ch[ch_p00, c_re  ]
            G = (line_curr_4ch[ch_p01, c_re  ] + line_curr_4ch[ch_p01, c_re-1] + \
                 line_curr_4ch[ch_p10, c_re  ] + line_prev_4ch[ch_p10, c_re  ]) * 0.25 
            R = (line_prev_4ch[ch_p11, c_re-1] + line_prev_4ch[ch_p11, c_re  ] + \
                 line_curr_4ch[ch_p11, c_re-1] + line_curr_4ch[ch_p11, c_re  ]) * 0.25
        else: # p01-site
            G =  line_curr_4ch[ch_p01, c_re  ]
            B = (line_curr_4ch[ch_p00, c_re  ] + line_curr_4ch[ch_p00, c_re+1]) * 0.5
            R = (line_prev_4ch[ch_p11, c_re  ] + line_curr_4ch[ch_p11, c_re  ]) * 0.5
        even_r_line[c_out] = R
        even_g_line[c_out] = G
        even_b_line[c_out] = B

    # --- 计算第二行输出 (奇数行, p10/p11 row) ---
    for c_out in range(W_out):
        c_re = c_out // 2 + 1
        if c_out % 2 == 0: # p10-site
            G =  line_curr_4ch[ch_p10, c_re  ]
            B = (line_curr_4ch[ch_p00, c_re  ] + line_next_4ch[ch_p00, c_re  ]) * 0.5
            R = (line_curr_4ch[ch_p11, c_re-1] + line_curr_4ch[ch_p11, c_re  ]) * 0.5
        else: # p11-site
            R =  line_curr_4ch[ch_p11, c_re  ]
            G = (line_curr_4ch[ch_p10, c_re  ] + line_curr_4ch[ch_p10, c_re+1] + \
                 line_curr_4ch[ch_p01, c_re  ] + line_next_4ch[ch_p01, c_re  ]) * 0.25
            B = (line_curr_4ch[ch_p00, c_re  ] + line_curr_4ch[ch_p00, c_re+1] + \
                 line_next_4ch[ch_p00, c_re  ] + line_next_4ch[ch_p00, c_re+1]) * 0.25
        odd_r_line[c_out] = R
        odd_g_line[c_out] = G
        odd_b_line[c_out] = B

cdef inline void _ccm_gamma_gather_rows(
    np.float32_t[:, :, ::1] rgb_line_buffer, # (2, 3, W_out)
    np.float32_t[:, :, ::1] final_img, 
    int r_out_start, int W_orig,
    np.float32_t[:, ::1] conversion_mtx,
    np.float32_t[::1] gamma_lut,
    float inv_clip_max_level):
    """
    在2行RGB缓冲上执行CCM和Gamma校正，并存入最终图像。
    params:
    rgb_line_buffer: 入参，2行RGB图，float32_t类型。大小(2, 3, W_out) (HCW格式)
    final_img: 出参，最终的RGB图像，float32_t类型。大小(H_orig, 3, W_orig) (HCW格式) 最终图像格式待定。
    W_orig: 入参，输出图像宽度，等于原始Bayer图像的宽度，int类型。
    conversion_mtx: 入参，3x3 CCM转换矩阵，float类型。
    gamma_lut: 入参，Gamma查找表，float类型。大小(65536,)
    inv_clip_max_level: 入参，clip_max_level的倒数，float类型。
    """
    cdef int c_out, r_idx, g_idx, b_idx
    cdef float r_val, g_val, b_val, r_ccm, g_ccm, b_ccm
    cdef int lut_max_index = gamma_lut.shape[0] - 1
    cdef float m00 = conversion_mtx[0, 0], m01 = conversion_mtx[0, 1], m02 = conversion_mtx[0, 2]
    cdef float m10 = conversion_mtx[1, 0], m11 = conversion_mtx[1, 1], m12 = conversion_mtx[1, 2]
    cdef float m20 = conversion_mtx[2, 0], m21 = conversion_mtx[2, 1], m22 = conversion_mtx[2, 2]
    cdef int r_offset

    # 循环处理两行
    # TODO: 我来做，后续进行SIMD friendly 排列。
    for r_offset in range(2):
        for c_out in range(W_orig):
            r_val = rgb_line_buffer[r_offset, 0, c_out] * inv_clip_max_level
            g_val = rgb_line_buffer[r_offset, 1, c_out] * inv_clip_max_level
            b_val = rgb_line_buffer[r_offset, 2, c_out] * inv_clip_max_level

            r_ccm = r_val * m00 + g_val * m01 + b_val * m02
            g_ccm = r_val * m10 + g_val * m11 + b_val * m12
            b_ccm = r_val * m20 + g_val * m21 + b_val * m22

            if r_ccm < 0: r_ccm = 0
            elif r_ccm > 1: r_ccm = 1
            if g_ccm < 0: g_ccm = 0
            elif g_ccm > 1: g_ccm = 1
            if b_ccm < 0: b_ccm = 0
            elif b_ccm > 1: b_ccm = 1

            r_idx = <int>(r_ccm * lut_max_index + 0.5)
            g_idx = <int>(g_ccm * lut_max_index + 0.5)
            b_idx = <int>(b_ccm * lut_max_index + 0.5)

            final_img[r_out_start + r_offset, 0, c_out] = <float>gamma_lut[r_idx]
            final_img[r_out_start + r_offset, 1, c_out] = <float>gamma_lut[g_idx]
            final_img[r_out_start + r_offset, 2, c_out] = <float>gamma_lut[b_idx]

cdef void _run_pipeline_gather(
    np.float32_t[:, :, ::1] raw_4ch,
    np.float32_t[:, :, ::1] final_img,

    np.float32_t[:, ::1] conversion_mtx,
    np.float32_t[::1] gamma_lut,
    int clip_max_level,
    bint pattern_is_bggr,

    np.float32_t[:, :, ::1] lines_buffer_4ch, # (3, 4, W_orig // 2 + 2)
    np.float32_t[:, :, ::1] rgb_line_buffer, # (2, 3, W_orig)
    ):
    """
    使用收集(Gather)模式的完整流水线。
    params:
    raw_4ch: 入参，原始4通道Bayer图像，float32_t类型。大小(H_orig, 4, W_orig) (HCW格式)
    final_img: 出参，最终的RGB图像，float32_t类型。大小(H_orig, 3, W_orig) (HCW格式)

    conversion_mtx: 入参，3x3 CCM转换矩阵，float类型。
    gamma_lut: 入参，Gamma查找表，float类型。大小(65536,)
    clip_max_level: 入参，最大ADC值，用于裁切，int类型。
    pattern_is_bggr: 入参，Bayer模式是否为BGGR，bool类型。

    lines_buffer_4ch: 入参，4通道Bayer图像行缓冲，float32_t类型。大小(3, 4, W_orig // 2 + 2) (HCW格式)
    rgb_line_buffer: 入参，2行RGB图，float32_t类型。大小(2, 3, W_orig) (HCW格式)
    """
    cdef int H_re = raw_4ch.shape[0]
    cdef int W_re = raw_4ch.shape[2]
    cdef int W_out = final_img.shape[2]
    cdef int r_re

    cdef float inv_clip_max_level = 1.0 / clip_max_level

    # --- 缓冲分配 ---
    # 建立并初始化行缓冲视图
    cdef float[:, ::1] line_prev_4ch = lines_buffer_4ch[0]
    cdef float[:, ::1] line_curr_4ch = lines_buffer_4ch[1]
    cdef float[:, ::1] line_next_4ch = lines_buffer_4ch[2]

    # --- 流水线启动 ---
    # 预加载前两行
    line_prev_4ch = raw_4ch[0]
    line_curr_4ch = raw_4ch[1]

    # --- 主循环 ---
    for r_re in range(1, H_re - 1):
        # 加载新的一行
        line_next_4ch = raw_4ch[r_re + 1]

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

        # 丢弃最早的行，准备接收下一行，旋转缓冲区指针一格。
        # 后续需要比较与直接索引取余哪个更好。
        line_prev_4ch = line_curr_4ch
        line_curr_4ch = line_next_4ch
        line_next_4ch = line_prev_4ch

# ------------------------------------------------------------------------------
# 2.2 SCATTER MODE IMPLEMENTATION
# ------------------------------------------------------------------------------

cdef inline void _debayer_scatter_row(
    const float [:, ::1] wb_4ch_row, # Shape (4, W_re)
    int r_re,
    float [:, :, ::1] final_img_padded, # Shape (H_orig + Y_PADDING, 3, W_orig + X_PADDING)
    int Y_PADDING,
    int X_PADDING   
    ):
    """
    处理单行已经白平衡的4通道数据，将其贡献扩散累加到输出缓冲。
    这是扩散模式最核心的计算函数。
    """
    cdef int W_re = wb_4ch_row.shape[1]
    cdef int c_re, y_p00, x_p00
    cdef float B, G1, G2, R
    
    # 定义物理通道索引 (p00, p01, p10, p11)
    # BGGR: B=p00, G1=p01, G2=p10, R=p11
    cdef int ch_p00 = 0, ch_p01 = 1, ch_p10 = 2, ch_p11 = 3

    # 遍历当前行的所有 2x2 像素块，值将会溢出到外围一圈（一行宽度）
    for c_re in range(W_re): # 此处必须完整遍历pad四通道的宽度，才能得到正确的边缘像素值。
        # 1. 读取4通道值
        B  = wb_4ch_row[ch_p00, c_re]
        G1 = wb_4ch_row[ch_p01, c_re]
        G2 = wb_4ch_row[ch_p10, c_re]
        R  = wb_4ch_row[ch_p11, c_re]

        # 2. 计算基准坐标
        y_p00 = 2 * (r_re - 1) + Y_PADDING // 2 # 双侧Padding
        x_p00 = 2 * (c_re - 1) + X_PADDING // 2 
        # 起始时刻：base=(1, 1), 但(1, 1)的值是错误的，正确的值从(3, 3)开始到(-4, -4)
        # y_p01 = y_p00
        # x_p01 = x_p00 + 1
        # y_p10 = y_p00 + 1
        # x_p10 = x_p00
        # y_p11 = y_p00 + 1
        # x_p11 = x_p00 + 1

        # 3. 扩散贡献 (BGGR Pattern)
        # 输出通道: 0=R, 1=G, 2=B
        
        # 首先实现naive版本，后续再进行SIMD优化。
        # --- R 通道贡献 ---
        # R-site (p11)
        final_img_padded[y_p00 + 1, 0, x_p00 + 1] += R # 自己
        # G-sites (p01, p10)
        final_img_padded[y_p00,     0, x_p00 + 1] += R * 0.5 # 同block p01
        final_img_padded[y_p00 + 1, 0, x_p00    ] += R * 0.5 # 同block p10
        final_img_padded[y_p00 + 1, 0, x_p00 + 2] += R * 0.5 # 右侧block p10
        final_img_padded[y_p00 + 2, 0, x_p00 + 1] += R * 0.5 # 下方block p01
        # B-site (p00)R
        final_img_padded[y_p00,     0, x_p00    ] += R * 0.25 # 同block p00
        final_img_padded[y_p00,     0, x_p00 + 2] += R * 0.25 # 右侧block p00
        final_img_padded[y_p00 + 2, 0, x_p00    ] += R * 0.25 # 下方block p00
        final_img_padded[y_p00 + 2, 0, x_p00 + 2] += R * 0.25 # 下方右侧block p00

        # --- G1 通道贡献 ---
        # G-site (p01)
        final_img_padded[y_p00,     1, x_p00 + 1] += G1 # 自己
        # R-site (p11)
        final_img_padded[y_p00 - 1, 1, x_p00 + 1] += G1 * 0.25 # 上方block p11
        final_img_padded[y_p00 + 1, 1, x_p00 + 1] += G1 * 0.25 # 同block p11
        # B-site (p00)
        final_img_padded[y_p00,     1, x_p00    ] += G1 * 0.25 # 同block p00
        final_img_padded[y_p00,     1, x_p00 + 2] += G1 * 0.25 # 右侧block p00

        # --- G2 通道贡献 ---
        # G-site (p10)
        final_img_padded[y_p00 + 1, 1, x_p00    ] += G2 # 自己
        # R-site (p11)
        final_img_padded[y_p00 + 1, 1, x_p00 - 1] += G2 * 0.25 # 左侧block p11
        final_img_padded[y_p00 + 1, 1, x_p00 + 1] += G2 * 0.25 # 同block p11
        # B-site (p00)
        final_img_padded[y_p00,     1, x_p00    ] += G2 * 0.25 # 同block p00
        final_img_padded[y_p00 + 2, 1, x_p00    ] += G2 * 0.25 # 下方block p00

        # --- B 通道贡献 ---
        # B-site (p00)
        final_img_padded[y_p00,     2, x_p00    ] += B # 自己
        # G-sites (p01, p10)
        final_img_padded[y_p00,     2, x_p00 - 1] += B * 0.5 # 左侧block p01
        final_img_padded[y_p00,     2, x_p00 + 1] += B * 0.5 # 同block p01
        final_img_padded[y_p00 - 1, 2, x_p00    ] += B * 0.5 # 上方block p10
        final_img_padded[y_p00 + 1, 2, x_p00    ] += B * 0.5 # 同block p10
        # R-site (p11)
        final_img_padded[y_p00 - 1, 2, x_p00 - 1] += B * 0.25 # 左侧block p11
        final_img_padded[y_p00 - 1, 2, x_p00 + 1] += B * 0.25 # 上方block p11
        final_img_padded[y_p00 + 1, 2, x_p00 - 1] += B * 0.25 # 左侧下方block p11
        final_img_padded[y_p00 + 1, 2, x_p00 + 1] += B * 0.25 # 同block p11

cdef inline void _process_CCM_gamma_scatter_rows(
    float [:, :, ::1] final_img_padded,
    int y_to_calculate,
    int W_padded,
    const float[3][3] ccm,
    const float[::1] gamma_lut,
    float inv_clip_max_level
    ):
    cdef int c_out, r_idx, g_idx, b_idx
    cdef float r_val, g_val, b_val, r_ccm, g_ccm, b_ccm
    cdef int lut_max_index = gamma_lut.shape[0] - 1
    cdef float m00 = ccm[0][0], m01 = ccm[0][1], m02 = ccm[0][2]
    cdef float m10 = ccm[1][0], m11 = ccm[1][1], m12 = ccm[1][2]
    cdef float m20 = ccm[2][0], m21 = ccm[2][1], m22 = ccm[2][2]

    cdef int r_offset, y_idx

    for r_offset in range(2):
        y_idx = y_to_calculate + r_offset
        for c_out in range(W_padded):
            r_val = final_img_padded[y_idx][0][c_out]
            g_val = final_img_padded[y_idx][1][c_out]
            b_val = final_img_padded[y_idx][2][c_out]

            r_ccm = (r_val * m00 + g_val * m01 + b_val * m02) * inv_clip_max_level
            g_ccm = (r_val * m10 + g_val * m11 + b_val * m12) * inv_clip_max_level
            b_ccm = (r_val * m20 + g_val * m21 + b_val * m22) * inv_clip_max_level

            r_ccm = clip_0_1(r_ccm)
            g_ccm = clip_0_1(g_ccm)
            b_ccm = clip_0_1(b_ccm)

            r_idx = <int>(r_ccm * lut_max_index + 0.5)
            g_idx = <int>(g_ccm * lut_max_index + 0.5)
            b_idx = <int>(b_ccm * lut_max_index + 0.5)

            final_img_padded[y_idx][0][c_out] = gamma_lut[r_idx]
            final_img_padded[y_idx][1][c_out] = gamma_lut[g_idx]
            final_img_padded[y_idx][2][c_out] = gamma_lut[b_idx]

cdef void _run_pipeline_scatter(
    float [:, :, ::1] raw_4ch,
    float [:, :, ::1] final_img_padded,

    const float r_gain, const float g_gain, const float b_gain,
    const float r_dBLC, const float g_dBLC, const float b_dBLC,
    const float[3][3] ccm,
    const np.float32_t [::1] gamma_lut,
    bint pattern_is_bggr,
    const int clip_max_level,
    int Y_PADDING, int X_PADDING,
    float[:, ::1] line_buf_4ch_wb, # (4, W_re)
    ):
    '''
    执行完整的流水线，包括行白平衡、扩散Debayer、CCM转换和gamma校正。
    
    params:
    raw_4ch: 输入的Bayer4通道图像。大小(H, 4, W) (HCW格式)
    final_img_padded: 输出的填充图像数据。大小(H+Y_PADDING, 3, W+X_PADDING)

    r_gain, g_gain, b_gain: 白平衡增益。
    r_dBLC, g_dBLC, b_dBLC: 额外黑电平。
    ccm: 颜色转换矩阵，以3x3的float数组形式传入。
    gamma_lut: gamma校正查找表，index应为int16范围(size=65536)。
    pattern_is_bggr: 布尔值，指示是否为BGGR模式。
    clip_max_level: 最大像素值。
    Y_PADDING, X_PADDING: 填充的Y和X方向总大小，两侧对称。这可能是一个优化点。

    line_buf_4ch_wb: 用于行白平衡的缓冲区，大小为(4, W_re)。
    '''
    cdef int H_re = raw_4ch.shape[0]
    cdef int W_re = raw_4ch.shape[2]
    cdef int W_padded = final_img_padded.shape[2]
    cdef int r_re, y_to_calculate
    cdef float inv_clip_max_level = 1.0 / clip_max_level

    for r_re in range(H_re): # 必须把所有padding都计算完，才能使内圈effective区域的结果正确。
        # 阶段 A: Inplace行白平衡
        _white_balance_row_inplace(raw_4ch[r_re], 
            r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, clip_max_level, pattern_is_bggr)

        # 阶段 B: 扩散Debayer
        _debayer_scatter_row(raw_4ch[r_re], r_re, final_img_padded, Y_PADDING, X_PADDING)

        # 阶段 C: 弹出并后处理 (Pop & Process)
        # 根据数据依赖，在处理完 r_re 后，起始于 2*r_re-3 的两行数据已就绪。
        y_to_calculate = (2 * r_re - 3) + Y_PADDING // 2
        _process_CCM_gamma_scatter_rows(
            final_img_padded, y_to_calculate, W_padded,
            ccm, gamma_lut, inv_clip_max_level
        )



# ==============================================================================
# 3. 顶层函数 (Top-Level Functions)
# ==============================================================================

cdef np.ndarray[np.float32_t, ndim=1] c_create_bt709_lut(int size=65536) noexcept:
    cdef np.ndarray[np.float32_t, ndim=1] lut = np.empty(size, dtype=np.float32)
    cdef int i
    cdef float linear_input_f
    for i in range(size):
        linear_input_f = <float>i / <float>(size - 1)
        if linear_input_f < 0.018:
            lut[i] = 4.5 * linear_input_f
        else:
            lut[i] = 1.099 * powf(linear_input_f, 0.45) - 0.099
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

    np.float32_t[:, :, ::1] raw_4ch, # (H_orig//2+2, 4, W_orig//2+2)
    float[:, ::1] line_buf_4ch_wb, # (4, W_orig//2+2), 行白平衡缓冲区
    ):
    """ V6版完整流水线 (Scatter Mode) """

    cdef float[3][3] ccm_data # 创建一个栈上空间

    # 根据条件，将 conversion_mtx 的数据“复制”到新的 ccm 中
    cdef int i, j
    if pattern_is_bggr:
        for i in range(3):
            for j in range(3):
                ccm_data[i][j] = conversion_mtx[i, j]
    else:
        for j in range(3):
            ccm_data[0][j] = conversion_mtx[2, j]
            ccm_data[1][j] = conversion_mtx[1, j]
            ccm_data[2][j] = conversion_mtx[0, j]
    
    # 步骤 1: 全局重排与Padding (不进行白平衡)
    _rearrange_and_pad(img, raw_4ch, black_level)

    # 步骤 2: 运行扩散流水线（逐行）
    _run_pipeline_scatter(
        raw_4ch, final_img_padded,
        r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, 
        ccm_data, gamma_lut,
        pattern_is_bggr, clip_max_level,
        Y_PADDING, X_PADDING,
        line_buf_4ch_wb
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
    ):
    """
    V6版完整流水线，编排所有计算步骤。
    """

    cdef float[:, ::1] ccm = conversion_mtx

    # 步骤 0: 准备CCM, 如果是BGGR，我们需要交换CCM的R和B列来匹配Debayer输出的RGB顺序
    if pattern_is_bggr:
        ccm = conversion_mtx
    else:
        ccm[0] = conversion_mtx[2]
        ccm[1] = conversion_mtx[1]
        ccm[2] = conversion_mtx[0]


    # 步骤 1: 全局重排与Padding
    _rearrange_and_pad(img, raw_4ch, black_level)

    # 步骤 2: 原地白平衡
    if pattern_is_bggr:
        _white_balance_inplace(raw_4ch, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, clip_max_level, 
        3, 1, 2, 0
        )
    else:
        _white_balance_inplace(raw_4ch, r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC, clip_max_level, 
        0, 1, 2, 3
        )

    # 步骤 3: 运行基于行缓冲的计算流水线 (当前为Gather模式)
    _run_pipeline_gather(raw_4ch, final_img, ccm, gamma_lut, clip_max_level, pattern_is_bggr,
        lines_buffer_4ch, rgb_line_buffer
    )


def raw_processing_cy_V6(img: np.ndarray,
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
    V6版 Cython图像处理流水线。
    使用全局重排+行缓冲策略以最大化性能。
    """
    # --- 参数与类型准备 ---
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

    # --- 内存分配 ---
    cdef int H_orig = c_img.shape[0]
    cdef int W_orig = c_img.shape[1]
    # 4通道中间缓冲，带padding，(H, C, W)格式。
    cdef np.ndarray[np.float32_t, ndim=3] raw_4ch = np.empty((H_orig // 2 + 2, 4, W_orig // 2 + 2), dtype=np.float32)

    if gamma != 'BT709':
        raise NotImplementedError("Only BT709 gamma is supported in the Cython V6 pipeline.")

    # --- 内存分配 (Gather) ---
    cdef np.ndarray[np.float32_t, ndim=3] lines_buffer_4ch_gather = np.empty((3, 4, W_orig // 2 + 2), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] rgb_line_buffer_gather = np.empty((2, 3, W_orig), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] final_img_gather = np.empty((H_orig, 3, W_orig), dtype=np.float32)
    # --- 内存分配 (Scatter) ---
    cdef int Y_PADDING = 6 # 这部分扩大我精心计算了，是这样的。
    cdef int X_PADDING = 6 
    cdef np.ndarray[np.float32_t, ndim=2] line_buf_4ch_scatter = np.empty((4, W_orig // 2 + 2), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=3] final_img_padded_scatter = np.zeros((H_orig + Y_PADDING, 3, W_orig + X_PADDING), dtype=np.float32)
    if mode == 'gather':
        # 进入 nogil 上下文执行所有计算
        cy_full_pipeline_v6_gather(c_img, final_img_gather, 
                                   black_level,
                                   r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC,
                                   pattern_is_bggr, clip_max_level,
                                   conversion_mtx, c_BT709_LUT,
                                   raw_4ch, lines_buffer_4ch_gather, rgb_line_buffer_gather
                                   )
        return np.ascontiguousarray(final_img_gather.transpose(0, 2, 1)) 

    elif mode == 'scatter':
        # print('scatter mode')
        cy_full_pipeline_v6_scatter(c_img, final_img_padded_scatter, 
                                    black_level,
                                    r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC,
                                    pattern_is_bggr, clip_max_level,
                                    conversion_mtx, c_BT709_LUT,
                                    Y_PADDING, X_PADDING,
                                    raw_4ch, line_buf_4ch_scatter
                                    )

        # 返回时切掉padding
        return np.ascontiguousarray(final_img_padded_scatter[Y_PADDING // 2: Y_PADDING//2 + H_orig, :, X_PADDING // 2: X_PADDING // 2 + W_orig].transpose(0, 2, 1))

    else:
        raise ValueError("Mode must be 'gather' or 'scatter'")
