### 相机颜色校正和其他Cython/C实现的辅助库

校正参数获取(利用ColorChecker24色卡): Correction.ipynb, 依赖colour-science包.
Raw正向ISP: raw_processing_V11_release. 无特殊依赖. Using CPU capabilities: AVX, AVX2.

Cython/C实现的ISP性能是一大卖点, 
对于测试使用的2448x2048x16bit @ i9-14900K DDR5 4800MT/s Dual-channel:
- Numpy naive 实现: ~400ms
- OpenCV in Python 实现: ~100ms
- Numba JIT 实现: ~33ms
- 最新Cython/C 实现: ~13ms

其他辅助库:
- PrecisionTimer: C实现的高精度C函数触发器. 目前的输入是C函数指针以及一个相机句柄(int), 适配Huateng Camera的SDK.
- unpack_12bit_raw: 将Huateng相机12bit Packed格式的Raw解成每像素uint16格式. Using CPU capabilities: SSE, SSE2, SSSE3. 1.1ms, 2448x2048 @ i9-14900K 5.4GHz, DDR5 4800MT/s Dual-channel.
    - Huateng 相机的12bit raw格式: 2个12bit像素存储在3个uint8中. byte0/2为pixel0/2 的高8bit, byte1 low 4bit: pixel0 的低4bit, byte1 high 4bit: pixel1 的低4bit.


代码包含详细算法注释.
