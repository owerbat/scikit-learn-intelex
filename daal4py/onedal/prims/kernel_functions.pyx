
# distutils: language = c++
#cython: language_level=2

# Import the Python-level symbols of numpy

# Import the C-level symbols of numpy

from cpython.ref cimport PyObject
from libcpp cimport bool

include "kernel_functions.pxi"

cdef class PyLinearKernelParams:
    cdef linear_kernel_params pt

    def __init__(self, scale, shift):
        self.pt.scale = scale
        self.pt.shift = shift

    @property
    def scale(self):
        return self.pt.scale

    @scale.setter
    def scale(self,val):
        self.pt.scale = val

    @property
    def shift(self):
        return self.pt.shift

    @shift.setter
    def shift(self,val):
        self.pt.shift = val


cdef class PyLinearKernelCompute:
    cdef linear_kernel_compute * thisptr

    def __cinit__(self, PyLinearKernelParams params):
        self.thisptr = new linear_kernel_compute(&params.pt)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, x, y):
        self.thisptr.compute(<PyObject *>x, <PyObject *>y)

    def get_values(self):
        return <object>self.thisptr.get_values()
