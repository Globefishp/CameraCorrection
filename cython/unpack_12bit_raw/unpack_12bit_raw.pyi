import numpy as np
from numpy.typing import NDArray

def unpack_12bit_to_16bit(
    packed_array: NDArray[np.uint8], 
    height: int, 
    width: int
) -> NDArray[np.uint16]:
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
    ...

def unpack_12bit_to_16bit_fast(
    packed_array: NDArray[np.uint8], 
    height: int, 
    width: int
) -> NDArray[np.uint16]:
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
    ...
