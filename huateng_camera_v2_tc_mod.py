# We intend to adapt IMX264 camera. Currently this file is not used in proj.
# 1. ROI
# 2. 12bit (linear storage?)
# 3. Color transfer matrix + AWB? More accurate color.
# 4. FPS control by soft trigger.

import numpy as np
import platform
from typing import Optional, Tuple
import ctypes # For memmove
from typing import Tuple

import mvsdk

FRAME_TIME = 10
GAIN = 1.0
TIMECODE_DTYPE = np.dtype("uint32") # Use numpy dtype for timecode
TIMECODE_BYTES = TIMECODE_DTYPE.itemsize # Get size in bytes from dtype
APPENDED_ROWS_FOR_TIMECODE = 1 # Number of extra rows to append for storing metadata

# TODO: 更换了相机，尝试使用软件触发来控制帧率。

def extract_tc_from_frames(
    combined_frames: np.ndarray,
    original_height: int,
    original_width: int,
    channels: int,
    timecode_dtype: np.dtype = TIMECODE_DTYPE,
    expected_appended_rows: int = APPENDED_ROWS_FOR_TIMECODE
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    从包含嵌入时间码的帧数据中提取原始图像和时间码。
    时间码数据附加在帧末尾，以小端字节序存储。
    如果无法提取时间码，则时间码部分返回 None。

    Args:
        combined_frames: 包含原始图像数据和附加时间码的 NumPy 数组 (n, H+appended, W, C)。
                         n 是帧的数量，H 是原始图像高度，W 是宽度，C 是通道数。
        original_height: 原始图像的高度。
        original_width: 原始图像的宽度。
        channels: 图像的通道数。
        timecode_dtype: 时间码的 NumPy 数据类型 (默认为 np.uint32)。

    Returns:
        一个元组，包含：
        - 原始图像数据 (np.ndarray)，形状为 (n, H, W, C) 或 (n, combined_H, W, C) 如果 combined_H < H。
        - 提取到的时间码 (Optional[np.ndarray])，形状为 (n,)，数据类型为 timecode_dtype，如果无法提取则为 None。
    """
    if combined_frames.ndim != 4 or combined_frames.shape[3] != channels:
        raise ValueError(f"Invalid combined_frames shape. Expected (n, H+appended, W, C), got {combined_frames.shape} with channels {channels}.")

    n_frames, combined_height, width, _ = combined_frames.shape
    timecode_bytes = timecode_dtype.itemsize

    # 始终尝试提取原始图像部分，即使高度可能不符合预期
    # 如果 combined_height 小于 original_height，我们仍然提取到 combined_height
    # 这意味着图像数据可能不完整，但函数会返回它所拥有的。
    # 调用者应该意识到这一点。
    actual_original_height_to_slice = min(original_height, combined_height)
    original_images = combined_frames[:, :actual_original_height_to_slice, :, :]

    if width != original_width:
        # 宽度不匹配是更严重的问题，通常表明数据源配置错误
        raise ValueError(f"Combined frame width ({width}) does not match original width ({original_width}).")

    if combined_height < original_height:
        print(f"Warning in extract_tc_from_frames: Combined frame height ({combined_height}) is less than original height ({original_height}). "
              f"Image data may be incomplete. No timecode can be extracted.")
        return original_images, None

    if combined_height == original_height:
        # print(f"Debug in extract_tc_from_frames: Combined frame height ({combined_height}) is equal to original height ({original_height}). "
        #       f"No appended data for timecode.")
        return original_images, None

    # At this point, combined_height > original_height, so there are appended rows.
    actual_appended_rows = combined_height - original_height
    if actual_appended_rows != expected_appended_rows:
        print(f"Warning in extract_tc_from_frames: Actual appended rows ({actual_appended_rows}) "
              f"does not match expected ({expected_appended_rows}). "
              f"Will attempt to read TC from the expected location in the first appended row.")
    
    # 计算每个完整帧（包括所有附加行）的总字节大小
    # total_bytes_per_combined_frame = combined_height * width * channels
    
    # 检查是否有足够的空间来读取一个时间码（从 offset_in_flat_frame 开始）
    # 我们需要至少 timecode_bytes 的数据在 original_height * width * channels 之后。
    # 附加区域的实际大小是 (combined_height - original_height) * width * channels
    bytes_in_first_appended_row_segment = original_width * channels # 假设时间码在第一行的开头
    if bytes_in_first_appended_row_segment < timecode_bytes:
        print(f"Warning in extract_tc_from_frames: The first appended row segment (width*channels = {bytes_in_first_appended_row_segment} bytes) "
              f"is too small to contain timecode ({timecode_bytes} bytes). Cannot extract timecode.")
        return original_images, None

    try:
        # 提取时间码
        # 时间码位于原始图像数据之后，即在 combined_frames[n, original_height, 0:timecode_bytes_in_row, 0]
        # 我们需要从每个帧的附加数据中 (索引 original_height:) 的开头提取 timecode_bytes。
        
        # 获取所有帧的所有附加数据
        appended_data = combined_frames[:, original_height:, :, :] # Shape: (n, appended_rows, W, C)
        
        # 将每个帧的所有附加数据展平，并只取前面的部分，确保不会超出 timecode_bytes
        # 将附加数据明确地转换为 uint8 类型视图，以便按字节处理
        appended_data_uint8_view = appended_data.view(np.uint8)

        # Reshape the uint8 view to (n_frames, total_appended_bytes_per_frame) to easily slice the first timecode_bytes
        appended_data_flat_uint8 = appended_data_uint8_view.reshape(n_frames, -1)

        # Slice the bytes for the timecode
        timecode_byte_batch = appended_data_flat_uint8[:, :timecode_bytes] # Shape: (n, timecode_bytes)

        # 将字节块转换为指定 dtype 的整数数组 (小端字节序)
        extracted_timecodes = timecode_byte_batch.view(timecode_dtype).ravel() # .ravel() to make it (n,)

        return original_images, extracted_timecodes

    except Exception as e:
        print(f"Error during timecode extraction in extract_tc_from_frames: {e}")
        return original_images, None

# TODO: Cython Implementation.
def decode_12bit_packed_to_16bit_numpy(packed_data: np.ndarray) -> np.ndarray:
    """
    Decode 12-bit packed data to 16-bit numpy array.

    Args:
        packed_data: Containing 12-bit packed data, dtype=unit8
                     the size must be a multiple of 3.

    Returns:
        A numpy array, dtype=np.uint16, lower 4 bit are padded with 0.
    """
    # Code Generated by Google Gemini 2.5 Pro, Modified by Haiyun Huang
    # 后记：别忘了数据有黑位（目前测量到Raw下是32）
    # 确保输入数据长度是3的倍数
    assert packed_data.size % 3 == 0, "Input data size must be a multiple of 3."

    # 将数据重塑为每行3个字节
    three_byte_chunks = packed_data.reshape(-1, 3).astype(np.uint16)

    # 提取像素
    byte0 = three_byte_chunks[:, 0]
    byte1 = three_byte_chunks[:, 1]
    byte2 = three_byte_chunks[:, 2]

    # 通过位运算解码
    # Packed Raw具体编码方式依靠猜。
    # 过曝欠曝实验，确定的是byte0和byte2是两个像素的高8位。byte1应该是两个像素的低4位拼接而成。
    # 拍摄尺子观察竖线实验(限制输出最后四位)：p0的低4位是b1低4位，p1低4位是b1高4位。

    pixel0 = (byte0 << 4) | (byte1 & 0x0F)
    pixel1 = (byte2 << 4) | (byte1 >> 4)
    # 顺序拼接，用于猜pack格式时观测数据。真实情况下由于操作太复杂，虽然直观，但不使用这种。
    # pixel0 = (byte0 << 4) | (byte1 >> 4)
    # pixel1 = (byte1 & 0x0F) << 8 | byte2

    # 将两个像素数组交错合并为一个
    decoded_data = np.empty(pixel0.size + pixel1.size, dtype=np.uint16)
    decoded_data[0::2] = pixel0
    decoded_data[1::2] = pixel1

    # 返回数据，可做必要的位操作，如限制最后4位输出。
    return decoded_data # & 0x000F


def enumerate_cameras():
    # 枚举相机
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return

    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))

    cams = []
    for i in map(lambda x: int(x), input("Select cameras: ").split()):
        cam = Camera(DevList[i])
        if cam.open():
            cams.append(cam)

class Camera(object):
    """
    华腾相机，本文件中的 Camera 类根据 open 函数的 tc 参数决定是否返回帧末尾追加时间码的图像。
    如果启用时间码，时间码以 uint32 小端字节序存储，存储在图像的最后一行（Height）。
    hibitdepth: 0: 8bit, 1: 12bit(Packed模式:实际占用1.5bpp, 需要想办法解码)
    """
    def __init__(self, 
                 DevInfo, 
                 exposure_time_ms: float = FRAME_TIME, 
                 gain: float = GAIN, 
                 tc: bool = False, 
                 hibitdepth: int = 0, 
                 **kwargs):
        super(Camera, self).__init__()
        self.DevInfo = DevInfo
        self.hCamera = 0
        self.cap = None # Camera capabilities
        self.pFrameBuffer = 0 # Pointer to the allocated frame buffer memory
        self.frames_captured = 0 # stats for debug

        self.target_exposure_time_ms = exposure_time_ms
        self.acutal_exposure_time_ms = None
        self.target_gain = gain
        self.actual_gain = None

        # 图像维度和缓冲区大小
        self.image_width = 0
        self.image_height = 0
        self.image_channels = 0
        self.bit_depth = hibitdepth  # 在SDK中被称为media_type，详见open时枚举。
        self.pixel_bytes = 1 if hibitdepth==0 else 2  # 每pixel占用pixel_bytes内存。
        self.actual_image_buffer_size = 0 # Size in bytes for HxWxC image data
        
        self.appended_rows = APPENDED_ROWS_FOR_TIMECODE
        self.new_frame_height = 0 # image_height + appended_rows
        self.total_allocated_buffer_size = 0 # 包含追加时间码的大小 (H+appended_rows)xWxC
        self._timecode_enabled = tc

    @property
    def width(self) -> int:
        return self.image_width

    @property
    def height(self) -> int:
        return self.image_height
    
    @property
    def channels(self) -> int:
        return self.image_channels

    @property
    def output_frame_height(self) -> int: # 包含附加行在内的缓冲区高度
        return self.new_frame_height

    @property
    def timecode_enabled(self) -> bool:
        """Returns whether timecode fusion is enabled."""
        return self._timecode_enabled
    
    @property
    def timecode_timebase(self) -> int:
        """Returns the timebase (denominator) for timecode. 1/timebase = time per tick."""
        return 10000 # 0.1ms

    @property
    def target_fps(self) -> float:
        """根据目标曝光时间返回目标 FPS。"""
        if self.target_exposure_time_ms > 0:
            return 1000.0 / self.target_exposure_time_ms
        else:
            # 如果曝光时间无效，返回默认值或引发错误
            print("Warning: Invalid exposure_time_ms (<= 0), cannot calculate target_fps.")
            return 0.0 # Or raise ValueError("Exposure time must be positive")
    @property
    def actual_fps(self) -> float:
        """根据实际曝光时间返回实际 FPS。"""
        if self.acutal_exposure_time_ms is not None and self.acutal_exposure_time_ms > 0:
            return 1000.0 / self.acutal_exposure_time_ms
        elif self.acutal_exposure_time_ms is None:
            # 如果实际曝光时间尚未可用，返回 0.0
            raise ValueError("Unexpected error in actual_fps: acutal_exposure_time_ms is "
                             "not initialized yet. Call open() before get actual_fps.")
        else:
            # 保留针对无效正曝光时间的 ValueError
            raise ValueError("Unexpected error in actual_fps: Actual exposure time must be positive")


    def open(self):
        if self.hCamera > 0:
            return True

        # 打开相机
        hCamera = 0
        try:
            hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
            return False

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(hCamera)

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 获取相机支持的位深
        print(f"Camera.open: Camera supports {cap.iMediaTypeDesc} pixel format(s)")
        # 枚举pixel format
        for i in range(cap.iMediaTypeDesc):
            print(f"Camera.open: Pixel format {cap.pMediaTypeDesc[i].iIndex}: {cap.pMediaTypeDesc[i].GetDescription()}")
        # SDK目前支持8/12/16; hibitdepth就当作是12bit。
        if self.bit_depth == 0:
            print("Camera.open: Using 8bit pixel format.")
            mvsdk.CameraSetMediaType(hCamera, 0)
        elif self.bit_depth == 1:
            print("Camera.open: Using 12bit pixel format.")
            mvsdk.CameraSetMediaType(hCamera, 1)

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(hCamera)
        self.cap = cap # 存储相机特性

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # ------- 分配buffer --------
        # 设置图像的原始维度
        self.image_width = cap.sResolutionRange.iWidthMax
        self.image_height = cap.sResolutionRange.iHeightMax
        self.image_channels = 1 if monoCamera else 3
        # 尽管12bit Packed是1.5Bpp，我们依然分配2Bytes，方便管理。
        
        self.actual_image_buffer_size = self.image_height * self.image_width \
                                        * self.image_channels * self.pixel_bytes


        if self._timecode_enabled:
            print(f"Camera.open: Original image HxWxC = {self.image_height}x{self.image_width}x{self.image_channels}")
            print(f"Camera.open: Appending {self.appended_rows} row(s) for metadata.")

            # 计算新的帧高度和总缓冲区大小
            self.new_frame_height = self.image_height + self.appended_rows
            self.total_allocated_buffer_size = self.new_frame_height * self.image_width \
                                                * self.image_channels * self.pixel_bytes

            # 检查增加后的附加区域是否足够
            appended_area_size = self.total_allocated_buffer_size - self.actual_image_buffer_size
            if appended_area_size < TIMECODE_DTYPE.itemsize:
                print(f"Error: Appended area size ({appended_area_size} bytes) is too small to store timecode ({TIMECODE_DTYPE.itemsize} bytes).")
                print("Please increase APPENDED_ROWS_FOR_TIMECODE or ensure image dimensions provide enough space.")
                return False

            print(f"Camera.open: New buffer H'xWxC = {self.new_frame_height}x{self.image_width}x{self.image_channels}")
            print(f"Camera.open: Allocating total buffer size = {self.total_allocated_buffer_size} bytes.")

            buffer_size_to_allocate = self.total_allocated_buffer_size
        else:
            # If timecode is not enabled, allocate only the original image buffer size
            print(f"Camera.open: Timecode disabled. Allocating original image buffer size: {self.actual_image_buffer_size} bytes.")
            self.new_frame_height = self.image_height # New frame height is just original height
            self.total_allocated_buffer_size = self.actual_image_buffer_size # Total allocated is just original size
            buffer_size_to_allocate = self.actual_image_buffer_size

        # 分配足够大的RGB buffer (图像 + 附加行 或 仅图像)
        pFrameBuffer = mvsdk.CameraAlignMalloc(buffer_size_to_allocate, 16)

        # (可选测试) 用模式填充附加行区域，以检查 SDK 对大于实际帧大小的缓冲区行为。
        # 这有助于验证 CameraImageProcess 只写入缓冲区头部的图像部分。
        # 仅在启用时间码时执行此填充
        if self._timecode_enabled:
            fill_pattern = 0xAA
            appended_area_offset = self.actual_image_buffer_size
            appended_area_size = self.total_allocated_buffer_size - self.actual_image_buffer_size # Recalculate for clarity
            if appended_area_size > 0:
                print(f"Camera.open: Filling appended area (offset: {appended_area_offset}, size: {appended_area_size} bytes) with pattern {hex(fill_pattern)} for testing.")
                ctypes.memset(pFrameBuffer + appended_area_offset, fill_pattern, appended_area_size)
        
        # 测试Raw模式下的StartBit
        start_bit = mvsdk.CameraGetRawStartBit(hCamera)
        print(f"Camera.open: Raw start bit = {start_bit}")
        mvsdk.CameraSetRawStartBit(hCamera, -1)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(hCamera, 0)

        # 手动曝光
        mvsdk.CameraSetAeState(hCamera, 0)
        # 设置曝光时间
        mvsdk.CameraSetExposureTime(hCamera, self.target_exposure_time_ms * 1000)
        # 读取实际的曝光时间
        self.acutal_exposure_time_ms = mvsdk.CameraGetExposureTime(hCamera) / 1000.0  # Convert to ms

        # 设置增益
        mvsdk.CameraSetAnalogGainX(hCamera, self.target_gain)
        # 读取实际的增益
        self.actual_gain = mvsdk.CameraGetAnalogGainX(hCamera)

        # # 切换为手动白平衡
        # mvsdk.CameraSetWbMode(hCamera, 0)
        # # 设置一次白平衡
        # mvsdk.CameraSetOnceWB(hCamera)

        # 重置时间戳
        mvsdk.CameraRstTimeStamp(hCamera)

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(hCamera)

        self.hCamera = hCamera
        self.pFrameBuffer = pFrameBuffer
        # self.cap is already set
        return True

    def close(self):
        if self.hCamera > 0:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = 0

        if self.pFrameBuffer != 0: # Check if buffer was allocated
            mvsdk.CameraAlignFree(self.pFrameBuffer)
            self.pFrameBuffer = 0

    def grab_raw(self) -> Optional[np.ndarray]:
        '''
        Raw格式输出，暂不支持timecode。
        '''
        if self.hCamera == 0 or self.pFrameBuffer == 0:
            print("Error: Camera not opened or buffer not allocated.")
            if self._timecode_enabled:
                return None # Or return None, None if changing return type
            else:
                return None

        hCamera = self.hCamera
        pFrameBuffer = self.pFrameBuffer

        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 60000)
            # mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            # Copy RawData to a numpy array for inspection.
            # Image size before debayer is HxWx(Bpp), for 12bit packed, Bpp=1.5
            if self.bit_depth == 0:
                raw_data_ctypes = (mvsdk.c_ubyte * (self.image_width * self.image_height)).from_address(pRawData)
            elif self.bit_depth == 1: # 12bit = 1.5Bpp
                raw_data_ctypes = (mvsdk.c_ubyte * (self.image_width * self.image_height * 3 // 2)).from_address(pRawData)

            raw_data_np = np.frombuffer(raw_data_ctypes, dtype=np.uint8).copy()
            # Copy for further processing
            
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
            # print(raw_data_np.shape)
            if self.bit_depth == 1:
                raw_data_np = decode_12bit_packed_to_16bit_numpy(raw_data_np)
            
            raw_data_np = raw_data_np.reshape(self.image_height, self.image_width)
            # print(raw_data_np.shape, raw_data_np.dtype)

            return raw_data_np

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"Camera GetImageBuffer/Process failed ({e.error_code}): {e.message}")
            if self._timecode_enabled:
                return None # Or return None, None if changing return type
            else:
                print(f'grab_raw error: {e.message}, returning None...')
                return None
                

    def grab(self) -> Optional[np.ndarray]:
        '''
        抓取一帧。
        如果 open 时 tc=True，则将时间码嵌入预分配缓冲区的附加行中，
        并返回整个缓冲区的 NumPy 数组视图，重塑为 (H+appended, W, C)。
        时间码 (uint32) 写入附加行区域的开头，little-endian。
        时间码单位为 0.1ms，最大能记录119h。
        如果 open 时 tc=False，则返回原始图像数据 (H, W, C)。
        '''
        if self.hCamera == 0 or self.pFrameBuffer == 0:
            print("Error: Camera not opened or buffer not allocated.")
            if self._timecode_enabled:
                return None # Or return None, None if changing return type
            else:
                return None

        hCamera = self.hCamera
        pFrameBuffer = self.pFrameBuffer

        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 60000)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            if platform.system() == "Windows":
                # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
                # linux下直接输出正的，不需要上下翻转
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

            if self._timecode_enabled:
                # 将时间码嵌入保留空间 (附加行的开头)
                timecode_value = FrameHead.uiTimeStamp # 这是 uint32
                timecode_as_bytes = timecode_value.to_bytes(TIMECODE_BYTES, byteorder='little')

                # 附加行数据起始位置的偏移量
                timecode_write_offset = self.actual_image_buffer_size

                # 写入时间码字节流
                ctypes.memmove(pFrameBuffer + timecode_write_offset, timecode_as_bytes, TIMECODE_BYTES)

                # Create a NumPy view of the entire buffer (image data + appended row(s) with timecode)
                combined_data_ctypes = (mvsdk.c_ubyte * self.total_allocated_buffer_size).from_address(pFrameBuffer)
                combined_frame_np_flat = np.frombuffer(combined_data_ctypes, dtype=np.uint8)

                # Reshape to (H+appended, W, C)
                output_frame = combined_frame_np_flat.reshape((self.new_frame_height, self.image_width, self.image_channels))

                self.frames_captured += 1
                return output_frame
            else:
                # If timecode is not enabled, return the original image data
                frame_data = (mvsdk.c_ubyte * self.actual_image_buffer_size).from_address(pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((self.image_height, self.image_width))

                self.frames_captured += 1
                return frame


        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"Camera GetImageBuffer/Process failed ({e.error_code}): {e.message}")
            if self._timecode_enabled:
                return None # Or return None, None if changing return type
            else:
                return None
