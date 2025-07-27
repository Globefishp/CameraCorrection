from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys


# 对于GCC/Clang编译器，使用-fopenmp
# 对于MSVC (Windows), 使用/openmp
if sys.platform == 'win32':
    compile_args = ['/openmp']
    link_args = []
else:
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']

extensions = [
    Extension(
        "hello",
        ["hello.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++",
        include_dirs=[numpy.get_include()] # 包含numpy头文件
    )
]

setup(
    ext_modules=cythonize(
        extensions,
    )
)
