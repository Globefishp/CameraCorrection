import numpy as np
import mvsdk
from huateng_camera_v2_tc_mod import Camera
from raw_processing import raw_processing
from raw_processing_jit import raw_processing_jit
from raw_processing_jit_V2 import raw_processing_jit_V2
from raw_processing_jit_V3 import raw_processing_jit_V3
from raw_processing_jit_V3_1 import raw_processing_jit_V3_1
from raw_processing_jit_V4 import raw_processing_jit_V4
from raw_processing_jit_V5 import raw_processing_jit_V5
from raw_processing_cy import raw_processing_cy
from raw_processing_cy_V2 import raw_processing_cy_V2
from raw_processing_cy_V2_unroll import raw_processing_cy_V2_unroll
from raw_processing_cy_V3 import raw_processing_cy_V3
from raw_processing_cy_V4 import RawV4Processor
from raw_processing_cy_V5 import raw_processing_cy_V5
from raw_processing_cy_V6 import raw_processing_cy_V6
import matplotlib.pyplot as plt
import time
import cProfile
import pstats

current_jit_func = raw_processing_cy_V6
current_jit_func_name = 'raw_processing_jit_V6'

EXPOSURE_TIME = 10 # ms
correction_info = np.load('./correction_results.npy', allow_pickle=True).item()
correction_info['wb_params'] = (1.87217887201, 1.27358336204, 1.0, -16.2625453031, -13.099179932, 0.0)

XYZ_TO_SRGB = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660,  1.8760108,  0.0415560],
                        [ 0.0556434, -0.2040259,  1.0572252]])

DevList = mvsdk.CameraEnumerateDevice()
mycam = Camera(DevList[0], EXPOSURE_TIME, gain=1, hibitdepth=1)
mycam.open()

img = mycam.grab_raw()

mycam.close()
if current_jit_func_name == 'raw_processing_cy_V4':
    processor = RawV4Processor(black_level=32,
                               ADC_max_level=4096,
                               bayer_pattern='BGGR',
                               wb_params=correction_info['wb_params'],
                               fwd_mtx=correction_info['fwd_mtx'],
                               render_mtx=XYZ_TO_SRGB,
                               gamma='BT709',
                               )
    srgb_img = processor.process(img)
elif current_jit_func_name == 'raw_processing_cy_V6':
    srgb_img = current_jit_func(img, 
                            black_level=32, 
                            ADC_max_level=4096,
                            bayer_pattern='BGGR',
                            wb_params=correction_info['wb_params'],
                            fwd_mtx=correction_info['fwd_mtx'],
                            render_mtx=XYZ_TO_SRGB,
                            gamma='BT709',
                            mode='gather'
                            )
else:
    srgb_img = current_jit_func(img, 
                            black_level=32, 
                            ADC_max_level=4096,
                            bayer_pattern='BGGR',
                            wb_params=correction_info['wb_params'],
                            fwd_mtx=correction_info['fwd_mtx'],
                            render_mtx=XYZ_TO_SRGB,
                            gamma='BT709',
                            )

print(f'Img value range:[{srgb_img.max(), srgb_img.min()}]')
print(f'Img size:{srgb_img.shape}')
# Save img using matplotlib
# plt.imsave('srgb_img.png', srgb_img)

# 2. 多次运行并记录时间
num_runs = 50
run_times = []
timings_total = np.zeros(4)
print(f"\n--- 运行 {num_runs} 次 {current_jit_func_name} 函数并记录时间 ---")
for _ in range(num_runs):
    if current_jit_func_name == 'raw_processing_cy_V4':
        start_time = time.perf_counter()
        processor.process(img)
        end_time = time.perf_counter()
    elif current_jit_func_name == 'raw_processing_cy_V6':
        start_time = time.perf_counter()
        current_jit_func(img, 
                         black_level=32, 
                         ADC_max_level=4096,
                         bayer_pattern='BGGR',
                         wb_params=correction_info['wb_params'],
                         fwd_mtx=correction_info['fwd_mtx'],
                         render_mtx=XYZ_TO_SRGB,
                         gamma='BT709',
                         mode='gather'
                        )
        end_time = time.perf_counter()
    else:
        start_time = time.perf_counter()
        current_jit_func(img,
                       black_level=32,
                       ADC_max_level=4096,
                       bayer_pattern='BGGR',
                       wb_params=correction_info['wb_params'],
                       fwd_mtx=correction_info['fwd_mtx'],
                       render_mtx=XYZ_TO_SRGB,
                       gamma='BT709',
                       )
        end_time = time.perf_counter()
    # timings_total += timings
    run_times.append(end_time - start_time)

average_time = sum(run_times) / num_runs
# timings_average = timings_total / num_runs
std_dev = np.std(run_times)

print(f"\n--- {current_jit_func_name} 函数运行时间 (avg. ± std) ({num_runs} 次): {average_time*1000:.3f} ± {std_dev*1000:.3f} 毫秒 ---")
# print("\n--- 性能计时结果 (秒) ---")
# print(f"白平衡填充 (WB_fill_time): {timings_average[0]:.6f} s")
# print(f"去马赛克 (Demosaic_time): {timings_average[1]:.6f} s")
# print(f"色彩矩阵与Gamma (CCM_Gamma_time): {timings_average[2]:.6f} s")
# print(f"总处理时间 (Total_time): {timings_average[3]:.6f} s")


# 3. 详细性能剖析并直接打印结果
print(f"\n--- {current_jit_func_name} 函数详细性能剖析结果 ---")
profiler = cProfile.Profile()
if current_jit_func_name == 'raw_processing_cy_V4':
    profiler.enable()
    srgb_img = processor.process(img)
elif current_jit_func_name == 'raw_processing_cy_V6':
    profiler.enable()
    srgb_img = current_jit_func(img, 
                            black_level=32, 
                            ADC_max_level=4096,
                            bayer_pattern='BGGR',
                            wb_params=correction_info['wb_params'],
                            fwd_mtx=correction_info['fwd_mtx'],
                            render_mtx=XYZ_TO_SRGB,
                            gamma='BT709',
                            mode='gather'
                            )
else:
    profiler.enable()
    current_jit_func(img,
                   black_level=32,
                   ADC_max_level=4096,
                   bayer_pattern='BGGR',
                   wb_params=correction_info['wb_params'],
                   fwd_mtx=correction_info['fwd_mtx'],
                   render_mtx=XYZ_TO_SRGB,
                   gamma='BT709',
                   )
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats(15) # 打印前15个耗时最多的函数

plt.imshow(srgb_img)
plt.show()
