# file: setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import platform

# --- Windows/MSVC specific configuration ---
if platform.system() == "Windows":
    print("Configuring for Windows/MSVC build.")
    extra_compile_args = ['/O2', '/EHsc'] # Standard optimization and C++ exception handling model
    extra_link_args = [] # MSVC links kernel32.lib etc. by default
else: # Linux, macOS, etc.
    print("Configuring for POSIX build (Linux/macOS).")
    extra_compile_args = ['-O3', '-std=c99']
    extra_link_args = ['-lrt', '-lpthread']

extensions = [
    Extension(
        "PrecisionTimer",
        sources=[
            "PrecisionTimer.pyx",
            "timer_core.c"
        ],
        libraries=["Winmm"] if platform.system() == "Windows" else [],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    ext_modules=cythonize(extensions),
    package_data={
        "PrecisionTimer": ["*.pyi"],
    }
)
