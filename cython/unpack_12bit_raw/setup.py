from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# MSVC compiler flags
extra_compile_args = ['/O2', '/arch:AVX2', '/FAs', '/Qvec-report:2']

# Define the extension module
ext_modules = [
    Extension(
        "unpack_12bit_raw",
        sources=[
            "unpack_12bit_raw.pyx",
            "core_intrinsic_V2.c"
        ],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        language="c"
    )
]

setup(
    name="unpack_12bit_raw",
    package_data={"unpack_12bit_raw": ["*.pyi"]}, # Include all .pyi files in the package
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': "3"}
    ),
)
