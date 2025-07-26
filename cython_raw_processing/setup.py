from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# 对于GCC/Clang编译器，使用-fopenmp
# 对于MSVC (Windows), 使用/openmp
# compile_args = ['-fopenmp']
# link_args = ['-fopenmp']

# 如果在Windows上
import sys
if sys.platform == 'win32':
    compile_args = ['/openmp']
    link_args = []


extensions = [
    Extension(
        "hello",
        ["build_raw_processing/hello.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++",
        include_dirs=[numpy.get_include()] # 包含Numpy头文件
    )
]

setup(
    ext_modules=cythonize(extensions)
)