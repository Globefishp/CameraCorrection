# distutils: language = c
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython

# Import the C function declaration from the header file
cdef extern from "core_intrinsic.h":
    void unpack_12bit_raw(unsigned char *src, unsigned short *dst, int width)

def unpack_12bit_to_16bit(packed_array, int height, int width):
    """
    Unpacks a 1D or 2D numpy array of 12-bit packed data into a 
    2D 16-bit numpy array.

    Args:
        packed_array: A 1D or 2D, C-contiguous numpy array of uint8 type,
                      representing the 12-bit packed data.
        height (int): The height of the target image.
        width (int): The width of the target image.

    Returns:
        A new 2D numpy array (height x width) of uint16 type containing 
        the unpacked pixel data.
    
    Raises:
        ValueError: If the input array's total size does not match the 
                    expected size from height and width, or if the width
                    is not an even number.
    """
    if not isinstance(packed_array, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    # The C function requires width to be even.
    if width % 2 != 0:
        raise ValueError("Image width must be an even number.")

    cdef int num_pixels = height * width
    cdef int expected_packed_size = (num_pixels * 3) // 2

    if packed_array.size != expected_packed_size:
        raise ValueError(f"Input array size {packed_array.size} does not match "
                         f"the expected size {expected_packed_size} for a "
                         f"{height}x{width} 12-bit image.")

    # Ensure the input array is 1D and C-contiguous for pointer access
    # np.ravel creates a view if possible, avoiding a copy.
    cdef np.ndarray[np.uint8_t, ndim=1, mode='c'] flat_packed_array
    flat_packed_array = np.ascontiguousarray(np.ravel(packed_array), dtype=np.uint8)

    # Create the 1D output array first
    cdef np.ndarray[np.uint16_t, ndim=1, mode='c'] unpacked_array_1d
    unpacked_array_1d = np.empty(num_pixels, dtype=np.uint16)

    # Get pointers to the data buffers of the numpy arrays
    cdef unsigned char* src_ptr = &flat_packed_array[0]
    cdef unsigned short* dst_ptr = &unpacked_array_1d[0]

    # Call the C function to perform the unpacking
    unpack_12bit_raw(src_ptr, dst_ptr, num_pixels)

    # Reshape the 1D result into a 2D image and return it
    return unpacked_array_1d.reshape((height, width))


@cython.boundscheck(False)
@cython.wraparound(False)
def unpack_12bit_to_16bit_fast(np.ndarray[np.uint8_t, ndim=1, mode='c'] packed_array, int height, int width):
    """
    Unpacks a 1D numpy array of 12-bit packed data into a 2D 16-bit numpy array.
    
    This is a high-performance version that skips all input validation.
    The caller is responsible for ensuring that:
    - packed_array is a 1D, C-contiguous numpy array of uint8 type.
    - Its size is exactly (height * width * 3) / 2.
    - width is an even number.

    Args:
        packed_array: A 1D or 2D, C-contiguous numpy array of uint8 type,
                      representing the 12-bit packed data.
        height (int): The height of the target image.
        width (int): The width of the target image.

    Returns:
        A new 2D numpy array (height x width) of uint16 type containing 
        the unpacked pixel data.
    """
    cdef int num_pixels = height * width
    
    # Create the output numpy array for the 16-bit data
    cdef np.ndarray[np.uint16_t, ndim=1, mode='c'] unpacked_array_1d = np.empty(num_pixels, dtype=np.uint16)

    # Get pointers to the data buffers of the numpy arrays
    cdef unsigned char* src_ptr = &packed_array[0]
    cdef unsigned short* dst_ptr = &unpacked_array_1d[0]

    # Call the C function to perform the unpacking
    unpack_12bit_raw(src_ptr, dst_ptr, num_pixels)

    # Reshape the 1D result into a 2D image and return it
    return unpacked_array_1d.reshape((height, width))
