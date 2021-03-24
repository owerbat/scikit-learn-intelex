
from libcpp.string cimport string

cdef extern from "data/utils.h":
    cdef string to_std_string(PyObject * o) except +


cdef extern from "svm/svm_py.h" namespace "oneapi::dal::python":
    ctypedef struct classification_svm_params:
        string method
        string kernel
        int class_count
        double c
        double accuracy_threshold
        int max_iteration_count
        double cache_size
        bool shrinking
        double shift
        double scale
        int degree
        double sigma

    cdef cppclass classification_svm_train:
        classification_svm_train(classification_svm_params *) except +
        void train(PyObject * data, PyObject * labels, PyObject * weights) except +
        int get_support_vector_count()  except +
        PyObject * get_support_vectors() except +
        PyObject * get_support_indices() except +
        PyObject * get_coeffs() except +
        PyObject * get_biases() except +

    cdef cppclass classification_svm_infer:
        classification_svm_infer(classification_svm_params *) except +
        void infer(PyObject * data, PyObject * support_vectors, PyObject * coeffs, PyObject * biases) except +
        PyObject * get_labels() except +
        PyObject * get_decision_function() except +

