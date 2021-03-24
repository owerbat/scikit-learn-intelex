
# distutils: language = c++
#cython: language_level=2

# Import the Python-level symbols of numpy

# Import the C-level symbols of numpy
cimport numpy as npc


from cpython.ref cimport PyObject
from libcpp cimport bool

include "svm.pxi"

cdef class PySvmParams:
    cdef svm_params pt

    def __init__(self, method, kernel):
        self.pt.method = to_std_string(<PyObject *>method)
        self.pt.kernel = to_std_string(<PyObject *>kernel)

    @property
    def class_count(self):
        return self.pt.class_count

    @class_count.setter
    def class_count(self,val):
        self.pt.class_count = val

    @property
    def c(self):
        return self.pt.c

    @c.setter
    def c(self,val):
        self.pt.c = val

    @property
    def accuracy_threshold(self):
        return self.pt.accuracy_threshold

    @accuracy_threshold.setter
    def accuracy_threshold(self,val):
        self.pt.accuracy_threshold = val

    @property
    def max_iteration_count(self):
        return self.pt.max_iteration_count

    @max_iteration_count.setter
    def max_iteration_count(self,val):
        self.pt.max_iteration_count = val

    @property
    def sigma(self):
        return self.pt.sigma

    @sigma.setter
    def sigma(self,val):
        self.pt.sigma = val

cdef class PyClassificationSvmTrain:
    cdef svm_train[classification] * thisptr

    def __cinit__(self, PySvmParams params):
        self.thisptr = new svm_train[classification](&params.pt)

    def __dealloc__(self):
        del self.thisptr

    def train(self, data, labels, weights):
        self.thisptr.train(<PyObject *>data, <PyObject *>labels, <PyObject *>weights)

    def get_support_vectors(self):
        return <object>self.thisptr.get_support_vectors()

    def get_support_indices(self):
        return <object>self.thisptr.get_support_indices()

    def get_coeffs(self):
        return <object>self.thisptr.get_coeffs()

    def get_biases(self):
        return <object>self.thisptr.get_biases()


cdef class PyClassificationSvmInfer:
    cdef svm_infer[classification] * thisptr

    def __cinit__(self, PySvmParams params):
        self.thisptr = new svm_infer[classification](&params.pt)

    def __dealloc__(self):
        del self.thisptr

    def infer(self, data, support_vectors, coeffs, biases):
        self.thisptr.infer(<PyObject *>data, <PyObject *>support_vectors, <PyObject *>coeffs, <PyObject *>biases)

    def get_labels(self):
        return <object>self.thisptr.get_labels()

    def get_decision_function(self):
        return <object>self.thisptr.get_decision_function()
