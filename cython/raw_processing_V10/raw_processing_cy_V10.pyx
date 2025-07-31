# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

import cython
import numpy as np
cimport numpy as np
from typing import Tuple

# --- C-level imports from our own C code ---
cdef extern from "windows.h":
    ctypedef struct LARGE_INTEGER:
        long long QuadPart
    
    int QueryPerformanceFrequency(LARGE_INTEGER* lpFrequency)
    int QueryPerformanceCounter(LARGE_INTEGER* lpPerformanceCount)
    
cdef extern from "raw_processing_core.h":
    # Define a C-level boolean type for Cython to use
    ctypedef int bool

    ctypedef enum TimingSections:
        TIMING_PREPARE_BUFFER
        TIMING_DEBAYER
        TIMING_CCM_WB
        TIMING_GAMMA_WRITEMEM
        TIMING_COUNT
    
    void c_full_pipeline(
        const np.uint16_t* img,
        int H_orig,
        int W_orig,
        int black_level,
        float r_gain, float g_gain, float b_gain,
        float r_dBLC, float g_dBLC, float b_dBLC,
        bool pattern_is_bggr,
        float clip_max_level,
        const float* conversion_mtx,
        const float* gamma_lut,
        int gamma_lut_size,
        float* final_img,
        float* line_buffers,
        float* rgb_line_buffer,
        int* ccm_line_buffer,
        long long* timing_results
    )

# This function remains in Cython as it's a convenient way to create a NumPy array
# that will be passed to the C layer.
def c_create_bt709_lut(int size=65536):
    """
    Creates a lookup table (LUT) for BT.709 Gamma correction.
    This is a Python-facing function that returns a NumPy array.
    """
    cdef np.ndarray[np.float32_t, ndim=1] lut = np.empty(size, dtype=np.float32)
    cdef int i
    cdef float linear_input_f
    for i in range(size):
        linear_input_f = <float>i / <float>(size - 1)
        if linear_input_f < 0.018:
            lut[i] = <float>4.5 * linear_input_f
        else:
            # Using C's powf for potentially better performance
            lut[i] = <float>1.099 * pow(linear_input_f, 0.45) - <float>0.099
    return lut


cdef class RawV10Processor:
    # C-level attributes for fast access
    cdef int black_level, H_orig, W_orig
    cdef float clip_max_level
    cdef bool pattern_is_bggr
    cdef float r_gain, g_gain, b_gain
    cdef float r_dBLC, g_dBLC, b_dBLC
    # Store NumPy arrays as generic objects at the class level
    cdef object conversion_mtx
    cdef object gamma_lut
    
    # Pre-allocated buffers stored as objects
    cdef object line_buffers
    cdef object rgb_line_buffer
    cdef object ccm_line_buffer

    def __cinit__(self, int H_orig, int W_orig, int black_level, int ADC_max_level, str bayer_pattern,
                  tuple wb_params, np.ndarray fwd_mtx, np.ndarray render_mtx,
                  str gamma='BT709'):
        """
        Initializes the processor with all constant parameters and pre-allocates buffers.
        """
        self.H_orig = H_orig
        self.W_orig = W_orig
        self.black_level = black_level
        self.clip_max_level = <float>(ADC_max_level - black_level)
        self.pattern_is_bggr = (bayer_pattern == 'BGGR')
        
        self.r_gain, self.g_gain, self.b_gain, self.r_dBLC, self.g_dBLC, self.b_dBLC = wb_params

        cdef np.ndarray[np.float32_t, ndim=2] c_fwd_mtx = np.asarray(fwd_mtx, dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=2] c_render_mtx = np.asarray(render_mtx, dtype=np.float32)
        self.conversion_mtx = np.dot(c_render_mtx, c_fwd_mtx)

        if gamma == 'BT709':
            self.gamma_lut = c_create_bt709_lut(size=256)
        else:
            raise NotImplementedError(f"Gamma '{gamma}' is not supported.")
            
        cdef int W_padded = self.W_orig + 2
        self.line_buffers = np.empty((3, W_padded), dtype=np.float32)
        self.rgb_line_buffer = np.empty((3, self.W_orig), dtype=np.float32)
        self.ccm_line_buffer = np.empty((3, self.W_orig), dtype=np.int32)

    def process(self, np.ndarray img):
        """
        Processes a single Bayer RAW image frame by calling the external C function.
        """
        # Ensure input image is of the correct type and C-contiguous
        cdef np.ndarray[np.uint16_t, ndim=2, mode='c'] c_img = np.ascontiguousarray(img, dtype=np.uint16)
        
        # Allocate the final output buffer for each frame
        cdef np.ndarray[np.float32_t, ndim=3, mode='c'] final_float = np.empty((self.H_orig, self.W_orig, 3), dtype=np.float32)

        # Create typed memoryviews as local variables before passing to C
        cdef np.ndarray[np.float32_t, ndim=2, mode='c'] c_conversion_mtx = self.conversion_mtx
        cdef np.ndarray[np.float32_t, ndim=1, mode='c'] c_gamma_lut = self.gamma_lut
        cdef np.ndarray[np.float32_t, ndim=2, mode='c'] c_line_buffers = self.line_buffers
        cdef np.ndarray[np.float32_t, ndim=2, mode='c'] c_rgb_line_buffer = self.rgb_line_buffer
        cdef np.ndarray[int, ndim=2, mode='c'] c_ccm_line_buffer = self.ccm_line_buffer

        # Variables for timing
        cdef long long[<int>TIMING_COUNT] timing_results
        cdef LARGE_INTEGER freq
        QueryPerformanceFrequency(&freq)
        cdef double frequency = <double>freq.QuadPart

        cdef LARGE_INTEGER start_total, end_total
        QueryPerformanceCounter(&start_total)

        # Call the core C pipeline with pointers to the NumPy array data
        c_full_pipeline(
            &c_img[0, 0],
            self.H_orig, self.W_orig,
            self.black_level,
            self.r_gain, self.g_gain, self.b_gain,
            self.r_dBLC, self.g_dBLC, self.b_dBLC,
            self.pattern_is_bggr,
            self.clip_max_level,
            &c_conversion_mtx[0, 0],
            &c_gamma_lut[0],
            <int>c_gamma_lut.shape[0],
            &final_float[0, 0, 0],
            &c_line_buffers[0, 0],
            &c_rgb_line_buffer[0, 0],
            &c_ccm_line_buffer[0, 0],
            timing_results
        )
        
        QueryPerformanceCounter(&end_total)
        
        cdef double prepare_buffer_ms = (timing_results[<int>TIMING_PREPARE_BUFFER] * 1000.0) / frequency
        cdef double debayer_ms = (timing_results[<int>TIMING_DEBAYER] * 1000.0) / frequency
        cdef double ccm_wb_ms = (timing_results[<int>TIMING_CCM_WB] * 1000.0) / frequency
        cdef double gamma_write_mem_ms = (timing_results[<int>TIMING_GAMMA_WRITEMEM] * 1000.0) / frequency
        cdef double total_c_pipeline_ms = ((<double>end_total.QuadPart - start_total.QuadPart) * 1000.0) / frequency
        
        print(f"Prepare Buffer time: {prepare_buffer_ms:.3f} ms")
        print(f"Debayer time: {debayer_ms:.3f} ms")
        print(f"CCM & WB time: {ccm_wb_ms:.3f} ms")
        print(f"Gamma + Wirte Mem time: {gamma_write_mem_ms:.3f} ms")
        print(f"Total C Pipeline time: {total_c_pipeline_ms:.3f} ms")

        return final_float

def raw_processing_cy_V10(img: np.ndarray,
                         black_level: int,
                         ADC_max_level: int,
                         bayer_pattern: str,
                         wb_params: tuple,
                         fwd_mtx: np.ndarray,
                         render_mtx: np.ndarray,
                         gamma: str = 'BT709',
                         ) -> np.ndarray:
    """
    Python wrapper to instantiate and use the RawV8Processor.
    For processing sequences, the processor should be instantiated only once.
    """
    cdef int H_orig = img.shape[0]
    cdef int W_orig = img.shape[1]
    
    processor = RawV10Processor(H_orig, W_orig, black_level, ADC_max_level, bayer_pattern,
                               wb_params, fwd_mtx, render_mtx, gamma)
    return processor.process(img)
