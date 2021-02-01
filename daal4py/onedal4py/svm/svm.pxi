
from libcpp.string cimport string

# cdef extern from "oneapi/dal/table/common.hpp" namespace "oneapi::dal":
#     cdef cppclass table:
#         pass

# cdef extern from "data_management/data.h":
#     cdef table _input_to_onedal_table(PyObject * nda) except +


# cdef extern from "oneapi/dal/algo/svm.hpp" namespace "oneapi::dal::svm":
#     cdef cppclass descriptor[fptype]:
#         descriptor() except +
#         descriptor[fptype] set_c(double) except +
#         descriptor[fptype] set_accuracy_threshold(double) except +


# cdef extern from "oneapi/dal/algo/svm.hpp" namespace "oneapi::dal":
#     cdef void train[](const handle_t &handle, math_t *input,
#                          int n_rows, int n_cols, math_t *labels,
#                          const svmParameter &param,
#                          KernelParams &kernel_params,
#                          svmModel[math_t] &model,


cdef extern from "data_management/utils.h":
    cdef string to_std_string(PyObject * o) except +


cdef extern from "svm/svm_py.h" namespace "oneapi::dal::py_svm":
    cdef cppclass classification_svm:
        classification_svm(string, double, double, int, double, bool) except +
        void train_cpp(PyObject * data, PyObject * labels, PyObject * weights) except +
        int get_support_vector_count()  except +
        PyObject * get_support_vectors() except +
        PyObject * get_support_indices() except +

# etc