import numpy as np
cimport numpy as cnp # cimport用于访问C级别API
import cython
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free, abort

@cython.boundscheck(False)
@cython.wraparound(False)
def process_data_V1(double[:, ::1] data_array, double[::1] out_array):
    '''
    V1 version of process data
    '''
    cdef int i, j
    cdef Py_ssize_t height = data_array.shape[0] 
    cdef Py_ssize_t width = data_array.shape[1]
    # Py_ssize_t is like int, but designed to represent array shape 
    # with platform-dependent size(int32/int64)
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

def process_data_V2(double[:, ::1] data_array, double[::1] out_array):
    '''
    V2 version of process data
    '''
    cdef int i, j
    cdef Py_ssize_t height = data_array.shape[0]
    cdef Py_ssize_t width = data_array.shape[1]
    cdef double local_sum

    # data_array is 2D, out_array is 1D
    with nogil, parallel():
        for i in prange(height):
            local_sum = 0.0
            for j in range(width):
                local_sum = local_sum + data_array[i, j] * j 
                # local_sum += ... will identify local_sum as a reduction variable
                # which won't successfully compile: Cannot read reduction variable in loop body
                # this is a limit of prange. reduction variables should only be write in one way (*=, += etc.)
            out_array[i] = local_sum
            
    return out_array

def process_data_test(double[:, ::1] data_array, double[::1] out_array):
    '''
    V2 version of process data
    '''
    cdef int i, j
    cdef Py_ssize_t height = data_array.shape[0]
    cdef Py_ssize_t width = data_array.shape[1]

    # data_array is 2D, out_array is 1D
    with nogil, parallel():
        cdef double local_sum = 0.0
        for i in prange(height):
            for j in range(width):
                local_sum = local_sum + data_array[i, j] * j 
                # local_sum += ... will identify local_sum as a reduction variable
                # which won't successfully compile: Cannot read reduction variable in loop body
                # this is a limit of prange. reduction variables should only be write in one way (*=, += etc.)
            out_array[i] = local_sum
            
    return out_array
