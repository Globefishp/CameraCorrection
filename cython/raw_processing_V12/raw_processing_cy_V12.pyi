import numpy as np
from typing import Tuple, Any, Optional

# Type alias for NumPy arrays, a common practice in stub files.
NDArray = np.ndarray

# Type alias for the 6-element white balance parameters tuple to improve readability.
# (r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC)
WBParams = Tuple[float, float, float, float, float, float]

class RawV12Processor:
    """
    Cython class for processing RAW images.

    This class encapsulates the image processing pipeline parameters and
    pre-allocated buffers to efficiently process multiple frames.
    """
    def __init__(self,
                 H_orig: int,
                 W_orig: int,
                 black_level: int,
                 ADC_max_level: int,
                 bayer_pattern: str,
                 wb_params: WBParams,
                 fwd_mtx: NDArray,
                 render_mtx: NDArray,
                 gamma: str = 'BT709',
                 gamma_lut_size: int = 1024,
                 streaming: bool = True) -> None:
        """
        Initializes the processor. Corresponds to __cinit__ in Cython.

        Parameters
        ----------
        H_orig : int
            Height of the input image, used for buffer initialization.
        W_orig : int
            Width of the input image, used for buffer initialization.
        black_level : int
            The black level of the RAW sensor data in native camera units.
        ADC_max_level : int
            Maximum ADC value from the camera, used for highlight clipping.
        bayer_pattern : str
            The Bayer pattern of the sensor, e.g., 'BGGR' or 'RGGB'.
            Currently, only these two are supported.
        wb_params : WBParams
            A 6-element tuple of white balance parameters:
            (r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC).
            The first three are RGB channel gains, the last three are
            additional black level offsets for each channel.
        fwd_mtx : np.ndarray
            Forward matrix (3x3, float32) to convert from camera RGB to
            the CIE XYZ color space.
        render_mtx : np.ndarray
            Render matrix (3x3, float32) to convert from CIE XYZ to the
            output RGB color space (e.g., sRGB/BT.709 gamut).
        gamma : str, optional
            The gamma correction standard to apply.
            Currently, only 'BT709' is supported. Defaults to 'BT709'.
        gamma_lut_size : int, optional
            The size of the gamma lookup table, determining quantization
            precision. 1024 (10-bit) is often sufficient. Defaults to 1024.
        streaming : bool, optional
            Whether to enable streaming mode. If True, the processor will
            store the result to a pre-allocated buffer to imporve performance. 
            It will be overwritten in the next `.process()` call. 
            Defaults to False.
        """
        ...

    def process(self, img: NDArray, out: Optional[NDArray] = None) -> NDArray:
        """
        Processes a single Bayer RAW image frame.

        Parameters
        ----------
        img : np.ndarray
            The input 2D RAW image as a NumPy array (uint16).
            User should ensure the image size is *always* readable within declared
            size (`H_orig * W_orig`), or SegmentFault would occur.
        out : np.ndarray, optional
            Pre-allocated countinuous array of shape `(H_orig, W_orig, 3)` if you want 
            to store the result in place. If provided, `streaming` is ignored.

        Returns
        -------
        np.ndarray
            The processed 3D RGB image as a NumPy array (uint16). If streaming=True
            and out is None, the result buffer will be overwrite in the next `process`
            call.
        """
        ...

def raw_processing_cy_V12(img: NDArray,
                         black_level: int,
                         ADC_max_level: int,
                         bayer_pattern: str,
                         wb_params: WBParams,
                         fwd_mtx: NDArray,
                         render_mtx: NDArray,
                         gamma: str = 'BT709'
                         ) -> NDArray:
    """
    A wrapper function to instantiate and run the RAW processing pipeline.

    This function is convenient for single-image processing. For processing
    sequences, it is more efficient to create a RawV11Processor instance
    once and call its `process` method repeatedly.

    Parameters
    ----------
    img : np.ndarray
        The input 2D RAW image as a NumPy array (uint16).
    black_level : int
        The black level of the RAW sensor data in native camera units.
    ADC_max_level : int
        Maximum ADC value from the camera, used for highlight clipping.
    bayer_pattern : str
        The Bayer pattern of the sensor, e.g., 'BGGR' or 'RGGB'.
    wb_params : WBParams
        A 6-element tuple of white balance parameters:
        (r_gain, g_gain, b_gain, r_dBLC, g_dBLC, b_dBLC).
    fwd_mtx : np.ndarray
        Forward matrix (3x3, float32) to convert from camera RGB to
        the CIE XYZ color space.
    render_mtx : np.ndarray
        Render matrix (3x3, float32) to convert from CIE XYZ to the
        output RGB color space (e.g., sRGB/BT.709 gamut).
    gamma : str, optional
        The gamma correction standard to apply. Defaults to 'BT709'.

    Returns
    -------
    np.ndarray
        The processed 3D RGB image as a NumPy array (uint16).
    """
    ...
