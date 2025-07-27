import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import argparse # Added import

# --- Argument parsing ---
# Disable default help to avoid conflict with setuptools' --help
parser = argparse.ArgumentParser(description='Build Cython modules.', add_help=False)
parser.add_argument('--module', type=str, default='raw_processing_cy_V2',
                    help='The name of the Cython module to compile (without .pyx extension).')

# Parse known arguments, keeping setuptools arguments separate
args, remaining_argv = parser.parse_known_args()

# Restore sys.argv with only the necessary arguments for setuptools
# sys.argv[0] is the script name itself.
# remaining_argv contains arguments like 'build_ext', '--inplace', etc.
sys.argv = [sys.argv[0]] + remaining_argv

# Use the parsed module name
module_name = args.module
source_file = f"{module_name}.pyx"

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
                        #   '/Zi',
                          ]
else:
    # Flags for GCC / Clang. -mavx2 is more explicit.
    extra_compile_args = ['-O3', '-ffast-math', '-march=native', '-fopenmp', '-mavx2']

extra_link_args = [
                #    '/DEBUG',
                   ]

# Create Extension object
# Use the dynamic module_name and source_file
ext_modules = [
    Extension(
        module_name, # Use the dynamic name
        [source_file], # Use the dynamic source file
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)

# Copy to parent directory
import shutil
# Update the copy path to use the dynamic module name
# TODO: automatically determine the compile name (platform, python version)
shutil.copy(f"{module_name}.cp313-win_amd64.pyd", f"../../{module_name}.cp313-win_amd64.pyd")
