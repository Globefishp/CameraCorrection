import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import argparse
import shutil

# --- Argument parsing ---
# Use a fixed module name for this specific setup script
module_name = 'raw_processing_cy_V10'
pyx_source_file = f"{module_name}.pyx"
c_source_file = "raw_processing_core.c"

# --- Build configuration ---
if sys.platform == 'win32':
    # Flags for MSVC / clang-cl. /arch:AVX2 is key for vectorization.
    extra_compile_args = ['/O2', '/fp:fast', '/arch:AVX2', '/Qvec-report:1', '/FAs']
    extra_link_args = []
else:
    # Flags for GCC / Clang.
    extra_compile_args = ['-O3', '-ffast-math', '-march=native', '-fopenmp', '-mavx2']
    extra_link_args = ['-fopenmp']

# --- Define the extension ---
# It now includes both the .pyx and .c source files
ext_modules = [
    Extension(
        module_name,
        [pyx_source_file, c_source_file],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c' # Specify that we are linking C code
    )
]

# --- Run setup ---
setup(
    ext_modules=cythonize(
        ext_modules,
        # annotate=True,
        compiler_directives={'language_level': "3"}
    )
)

# --- Post-build step: Copy the compiled module to the project root ---
print("Build complete. Copying compiled module to project root...")
try:
    # Find the compiled file. The exact name depends on the platform and Python version.
    build_dir = 'build'
    compiled_file_name = None
    for root, dirs, files in os.walk(build_dir):
        for file in files:
            if file.startswith(module_name) and file.endswith(('.pyd', '.so')):
                compiled_file_path = os.path.join(root, file)
                # Copy to the parent directory of the 'cython' folder
                destination_path = os.path.join('..', '..', file)
                shutil.copy(compiled_file_path, destination_path)
                print(f"Copied '{compiled_file_path}' to '{destination_path}'")
                # Also copy to the project root for convenience
                shutil.copy(compiled_file_path, os.path.join('..', '..', '..', file))
                print(f"Copied '{compiled_file_path}' to project root.")
                break
        if compiled_file_name:
            break
except Exception as e:
    print(f"Error copying file: {e}")
    print("You may need to manually copy the compiled .pyd/.so file from the 'build' directory.")
