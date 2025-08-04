# 本测试必须在CPU定频下有效!

import cProfile

import time
import mvsdk
import numpy as np

from unpack_12bit_raw import unpack_12bit_to_16bit_fast

try:
    DevList = mvsdk.CameraEnumerateDevice()
    hCamera = mvsdk.CameraInit(DevList[0], -1, -1)
    cap = mvsdk.CameraGetCapability(hCamera)

    # 判断是黑白相机还是彩色相机
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    # ------- 分配buffer --------
    # 设置图像的原始维度
    image_width = cap.sResolutionRange.iWidthMax
    image_height = cap.sResolutionRange.iHeightMax
    image_channels = 1 if monoCamera else 3
    # 尽管12bit Packed是1.5Bpp，我们依然分配2Bytes，方便管理。

    actual_image_buffer_size = image_height * image_width \
                                    * image_channels * 2


    # 分配足够大的RGB buffer (图像 + 附加行 或 仅图像)
    pFrameBuffer = mvsdk.CameraAlignMalloc(actual_image_buffer_size, 16)

    # 相机模式切换成连续采集
    mvsdk.CameraSetTriggerMode(hCamera, 0)

    # 手动曝光
    mvsdk.CameraSetAeState(hCamera, 0)
    # 设置曝光时间
    mvsdk.CameraSetExposureTime(hCamera, 10 * 1000)

    # 设置增益
    mvsdk.CameraSetAnalogGainX(hCamera, 1)

    # 重置时间戳
    mvsdk.CameraRstTimeStamp(hCamera)

    # 让SDK内部取图线程开始工作
    mvsdk.CameraPlay(hCamera)

    time_stat = []
    for i in range(1000):
        pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)

        raw_data_ctypes = (mvsdk.c_ubyte * (image_width * image_height * 3 // 2)).from_address(pRawData)

        raw_data_np = np.frombuffer(raw_data_ctypes, dtype=np.uint8)
        start_time = time.perf_counter_ns()
        raw_data_np = unpack_12bit_to_16bit_fast(raw_data_np, image_height, image_width)
        end_time = time.perf_counter_ns()
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
        time_stat.append(end_time - start_time)
finally:

    print(f'avg. time: {np.mean(time_stat)/1000_000:.3f} ms')
    print(f'std. time: {np.std(time_stat)/1000_000:.3f} ms')
    # 关闭相机
    mvsdk.CameraUnInit(hCamera)
    # 释放内存
    mvsdk.CameraAlignFree(pFrameBuffer)


