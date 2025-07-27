import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# --- 使用 clang-cl 在 Windows 上自动编译的配置 ---
if sys.platform == 'win32':
    pass
    # 将环境变量指向 clang-cl
    # os.environ['CC'] = 'clang-cl.exe'
    # os.environ['CXX'] = 'clang-cl.exe'
    # 重要！这样指定clang-cl并不会真的调用clang-cl，能够调用的方法有待研究。

    # 移除之前注入 --compiler=mingw32 的逻辑，因为我们现在使用 clang-cl 兼容 MSVC
    # if 'build_ext' in sys.argv:
    #     if not any(arg.startswith('--compiler') for arg in sys.argv):
    #         try:
    #             build_ext_idx = sys.argv.index('build_ext')
    #             sys.argv.insert(build_ext_idx + 1, '--compiler=mingw32')
    #         except ValueError:
    #             pass

# 移除 clang 的特定编译和链接参数，让 clang-cl 像 MSVC 一样工作
# msvc_version = "14.44.35207"
# sdk_version = "10.0.26100.0"

# clang_compile_args = [
#     '-O3',
#     '-march=native',
#     '-ffast-math',
#     '--target=x86_64-pc-windows-gnu',
#     '-fms-extensions',
#     fr'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\{msvc_version}\include',
#     fr'C:\Program Files (x86)\Windows Kits\10\include\{sdk_version}\ucrt'
# ]
# clang_link_args = [
#     fr'-LC:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\{msvc_version}\lib\x64',
#     fr'-LC:\Program Files (x86)\Windows Kits\10\lib\{sdk_version}\ucrt\x64'
# ]

# Define optimization flags
# For MSVC on Windows (used by clang-cl), use /O2 for speed and /fp:fast for floating point.
# For other compilers (like GCC/Clang on Linux), use -O3 and -ffast-math.
extra_compile_args = []
if sys.platform == 'win32':
    # Flags for MSVC / clang-cl. /arch:AVX2 is key for vectorization.
    extra_compile_args = ['/Ox', 
                          '/fp:fast', 
                          '/openmp', 
                          '/arch:AVX2',
                          ]
else:
    # Flags for GCC / Clang. -mavx2 is more explicit.
    extra_compile_args = ['-O3', '-ffast-math', '-march=native', '-fopenmp', '-mavx2']

name = "raw_processing_cy_V2"
# 创建 Extension 对象
ext_modules = [
    Extension(
        name,
        [f"{name}.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
    )
]

setup(
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)

# Copy to parent directory
import shutil
shutil.copy(f"{name}.cp313-win_amd64.pyd", f"../../{name}.cp313-win_amd64.pyd")
