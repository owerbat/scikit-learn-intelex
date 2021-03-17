
# distutils: language = c++
#cython: language_level=2

# Import the Python-level symbols of numpy

# Import the C-level symbols of numpy
cimport numpy as npc


from cpython.ref cimport PyObject
from libcpp cimport bool

include "svm.pxi"

npc.import_array()

cdef class PyClassificationSvm:
    cdef classification_svm * thisptr

    def __cinit__(self, kernel, double c, double accuracy_threshold, int max_iteration_count, double cache_size, bool shrinking):
        self.thisptr = new classification_svm(to_std_string(<PyObject *>kernel), c, accuracy_threshold, max_iteration_count, cache_size, shrinking)

    def __dealloc__(self):
        del self.thisptr

    def train(self, data, labels, weights):

        self.thisptr.train_cpp(<PyObject *>data, <PyObject *>labels, <PyObject *>weights)

    def get_support_vectors(self):
        return <object>self.thisptr.get_support_vectors()

    def get_support_indices(self):
        return <object>self.thisptr.get_support_indices()

    def get_coeffs(self):
        return <object>self.thisptr.get_coeffs()

    def get_bias(self):
        return self.thisptr.get_bias()

    def infer(self, data):
        self.thisptr.infer_cpp(<PyObject *>data)

    def get_labels(self):
        return <object>self.thisptr.get_labels()

    def get_decision_function(self):
        return <object>self.thisptr.get_decision_function()
