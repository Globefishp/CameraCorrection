import numpy as np
cimport numpy as cnp # cimport用于访问C级别API
import cython
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free, abort

def process_data_numba(double[:, ::1] data_array, double[::1] out_array):
    cdef int i, j
    cdef int height = data_array.shape[0]
    cdef int width = data_array.shape[1]
    cdef double* local_sum_ptr

    # data_array is 2D, out_array is 1D
    with nogil, parallel():
        local_sum_ptr = <double*> malloc(sizeof(double))
        if local_sum_ptr is NULL:
            # 如果内存分配失败，中止程序。在生产代码中可能有更好的错误处理。
            abort()
        try:
            for i in prange(height):
                local_sum_ptr[0] = 0.0
                for j in range(width):
                    local_sum_ptr[0] += data_array[i, j] * j
                out_array[i] = local_sum_ptr[0]
        finally:
            free(local_sum_ptr)
            
    return out_array
