import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import mvsdk
from huateng_camera_v2_tc_mod import Camera

class RawOpenCVProcessor:
    def __init__(self, H_orig, W_orig, black_level, ADC_max_level, bayer_pattern,
                 wb_params, fwd_mtx, render_mtx, gamma='BT709'):
        
        self.H_orig = H_orig
        self.W_orig = W_orig
        self.black_level = black_level
        self.ADC_max_level = ADC_max_level
        self.bayer_pattern = bayer_pattern
        self.wb_params = wb_params
        self.fwd_mtx = fwd_mtx
        self.render_mtx = render_mtx
        self.gamma = gamma

        # 预计算最终的颜色校正矩阵
        self.final_ccm = np.dot(self.render_mtx, self.fwd_mtx)

        # 预计算Gamma查找表
        if self.gamma == 'BT709':
            self.gamma_lut = self._create_bt709_lut()
        else:
            raise NotImplementedError(f"Gamma '{gamma}' is not supported.")

    def _create_bt709_lut(self, size=256):
        """
        创建BT.709 Gamma校正的查找表。
        """
        lut = np.zeros((size, 1), dtype=np.uint8)
        for i in range(size):
            x = i / (size - 1.0)
            if x < 0.018:
                y = 4.5 * x
            else:
                y = 1.099 * (x**0.45) - 0.099
            lut[i] = np.clip(y * 255.0, 0, 255).astype(np.uint8)
        return lut

    def process(self, img_raw, verbose=False):
        timings = {}
        start_total = time.perf_counter()

        # 1. 数据类型转换
        start_time = time.perf_counter()
        img_float = img_raw.astype(np.float32)
        timings['type_conversion'] = time.perf_counter() - start_time

        # 2. 黑电平扣除
        start_time = time.perf_counter()
        # 全局黑电平扣除
        img_float -= self.black_level

        # 分通道黑电平扣除 (根据BGGR模式)
        R_offset, G_offset, B_offset = self.wb_params[3], self.wb_params[4], self.wb_params[5]

        # BGGR 模式:
        # B G
        # G R
        # 奇数行奇数列 (0,0) -> B
        # 奇数行偶数列 (0,1) -> G
        # 偶数行奇数列 (1,0) -> G
        # 偶数行偶数列 (1,1) -> R

        # 提取通道并应用偏移
        img_float[0::2, 0::2] -= B_offset # B
        img_float[0::2, 1::2] -= G_offset # G
        img_float[1::2, 0::2] -= G_offset # G
        img_float[1::2, 1::2] -= R_offset # R

        # 裁剪到0
        cv2.threshold(img_float, 0, 0, cv2.THRESH_TOZERO, dst=img_float)
        timings['black_level_correction'] = time.perf_counter() - start_time

        # 3. 白平衡
        start_time = time.perf_counter()
        R_gain, G_gain, B_gain = self.wb_params[0], self.wb_params[1], self.wb_params[2]

        # 应用增益
        img_float[0::2, 0::2] *= B_gain # B
        img_float[0::2, 1::2] *= G_gain # G
        img_float[1::2, 0::2] *= G_gain # G
        img_float[1::2, 1::2] *= R_gain # R

        # 裁剪到ADC_max_level
        cv2.min(img_float, self.ADC_max_level, dst=img_float)
        timings['white_balance'] = time.perf_counter() - start_time

        # 4. 归一化与类型转换 (为Demosaic准备)
        start_time = time.perf_counter()
        # 将图像归一化到0-65535，并转换为uint16
        img_normalized_uint16 = (img_float / self.ADC_max_level * 65535.0).astype(np.uint16)
        timings['normalize_and_convert_to_uint16'] = time.perf_counter() - start_time

        # 5. 去马赛克
        start_time = time.perf_counter()
        # OpenCV的去马赛克函数需要uint8或uint16输入
        if self.bayer_pattern == 'BGGR':
            img_rgb_linear = cv2.cvtColor(img_normalized_uint16, cv2.COLOR_BAYER_BGGR2RGB)
        elif self.bayer_pattern == 'RGGB':
            img_rgb_linear = cv2.cvtColor(img_normalized_uint16, cv2.COLOR_BAYER_RGGB2RGB)
        elif self.bayer_pattern == 'GRBG':
            img_rgb_linear = cv2.cvtColor(img_normalized_uint16, cv2.COLOR_BAYER_GRBG2RGB)
        elif self.bayer_pattern == 'GBRG':
            img_rgb_linear = cv2.cvtColor(img_normalized_uint16, cv2.COLOR_BAYER_GBRG2RGB)
        else:
            raise ValueError("Unsupported bayer_pattern. Must be 'BGGR', 'RGGB', 'GRBG', or 'GBRG'.")
        timings['demosaicing'] = time.perf_counter() - start_time

        # 6. 颜色矩阵转换 (CCM)
        start_time = time.perf_counter()
        # 将图像转换为float32并归一化到[0, 1]以便进行矩阵乘法
        img_rgb_linear_float = img_rgb_linear.astype(np.float32) / 65535.0

        # 应用颜色矩阵 (使用cv2.transform)
        img_rgb_ccm = cv2.transform(img_rgb_linear_float, self.final_ccm)

        # 裁剪到[0, 1]
        cv2.threshold(img_rgb_ccm, 0, 0, cv2.THRESH_TOZERO, dst=img_rgb_ccm) # 裁剪下限
        cv2.min(img_rgb_ccm, 1.0, dst=img_rgb_ccm) # 裁剪上限
        timings['color_matrix_conversion'] = time.perf_counter() - start_time

        # 7. Gamma映射 (BT.709)
        start_time = time.perf_counter()
        if self.gamma == 'BT709':
            # 将图像归一化到 [0, 255] 并转换为 uint8
            img_for_gamma = (img_rgb_ccm * 255.0).astype(np.uint8)
            
            srgb_img_uint8 = cv2.LUT(img_for_gamma, self.gamma_lut)
            srgb_img = srgb_img_uint8.astype(np.float32) / 255.0 # 转换回 [0, 1] 浮点数
        else:
            srgb_img = img_rgb_ccm

        # 裁剪到[0, 1] (在LUT转换后可能不需要，但为了安全保留)
        cv2.threshold(srgb_img, 0, 0, cv2.THRESH_TOZERO, dst=srgb_img)
        cv2.min(srgb_img, 1.0, dst=srgb_img)
        timings['gamma_mapping'] = time.perf_counter() - start_time

        timings['total_processing_time'] = time.perf_counter() - start_total

        if verbose:
            print("\n--- OpenCV处理步骤耗时 (秒) ---")
            for step, duration in timings.items():
                print(f"{step}: {duration:.6f} s")

        return srgb_img, timings

if __name__ == '__main__':
    correction_info = np.load('./correction_results.npy', allow_pickle=True).item()
    correction_info['wb_params'] = (1.87217887201, 1.27358336204, 1.0, -16.2625453031, -13.099179932, 0.0)

    XYZ_TO_SRGB = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                            [-0.9692660,  1.8760108,  0.0415560],
                            [ 0.0556434, -0.2040259,  1.0572252]])

    EXPOSURE_TIME = 10 # ms
    # 尝试连接相机并抓取图像
    try:
        DevList = mvsdk.CameraEnumerateDevice()
        if not DevList:
            raise Exception("未找到相机设备。请确保相机已连接并驱动正常。")
        mycam = Camera(DevList[0], EXPOSURE_TIME, gain=1, hibitdepth=1)
        mycam.open()
        img = mycam.grab_raw()
        mycam.close()
    except Exception as e:
        print(f"相机操作失败: {e}")
        print("将使用一个模拟的随机图像进行处理。")
        # 创建一个模拟的uint16 BGGR图像用于测试
        height, width = 2048, 2448 # 示例尺寸
        img = np.random.randint(0, 4096, size=(height, width), dtype=np.uint16)
        # 模拟BGGR模式，确保像素值分布符合预期
        # 假设Bayer模式是BGGR
        # B G
        # G R
        img[0::2, 0::2] = np.random.randint(0, 4096, size=(height//2, width//2), dtype=np.uint16) # B
        img[0::2, 1::2] = np.random.randint(0, 4096, size=(height//2, width//2), dtype=np.uint16) # G
        img[1::2, 0::2] = np.random.randint(0, 4096, size=(height//2, width//2), dtype=np.uint16) # G
        img[1::2, 1::2] = np.random.randint(0, 4096, size=(height//2, width//2), dtype=np.uint16) # R


    print(f"输入图像尺寸: {img.shape}, 数据类型: {img.dtype}")

    # 实例化处理器
    processor = RawOpenCVProcessor(H_orig=img.shape[0],
                                   W_orig=img.shape[1],
                                   black_level=32,
                                   ADC_max_level=4096,
                                   bayer_pattern='BGGR',
                                   wb_params=correction_info['wb_params'],
                                   fwd_mtx=correction_info['fwd_mtx'],
                                   render_mtx=XYZ_TO_SRGB,
                                   gamma='BT709'
                                   )

    # 首次运行以获取结果并打印详细耗时
    srgb_img, _ = processor.process(img, verbose=True)

    print(f'处理后图像值范围:[{srgb_img.max():.4f}, {srgb_img.min():.4f}]')
    print(f'处理后图像尺寸:{srgb_img.shape}')

    # 2. 多次运行并记录时间
    num_runs = 10
    run_times = []
    print(f"\n--- 运行 {num_runs} 次 raw_processing_opencv 函数并记录时间 ---")
    for _ in range(num_runs):
        _, timings_run = processor.process(img, verbose=False)
        run_times.append(timings_run['total_processing_time'])

    average_time = sum(run_times) / num_runs
    std_dev = np.std(run_times)

    print(f"\n--- raw_processing_opencv 函数运行时间 (avg. ± std) ({num_runs} 次): {average_time*1000:.3f} ± {std_dev*1000:.3f} 毫秒 ---")

    # 显示图像
    plt.imshow(srgb_img)
    plt.title('Processed sRGB Image (OpenCV)')
    plt.axis('off')
    plt.show()
