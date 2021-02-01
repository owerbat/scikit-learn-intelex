
# distutils: language = c++
#cython: language_level=2

# Import the Python-level symbols of numpy
import numpy as np

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

    def get_support_indices(self):
        return <object>self.thisptr.get_support_indices()

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y

class SVC(ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0,
                 coef0=0.0, tol=1e-3, cache_size=200.0, max_iter=-1,
                 class_weight=None, algorithm='thunder'):

        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.algorithm = algorithm

    def _get_kernel_params(self, x):
        if self.kernel == 'linear':
            if self.dtype == np.float32:
                pass
                # cdef kernel_function_linear[float] kernel
                # return kernel
        # elif self.kernel == 'rbf':
            # kf = daal4py.kernel_function_rbf(
            # fptype=X_fptype, method=method, sigma=sigma_value)
        else:
            raise ValueError(
                f"_daal4py_fit received unexpected kernel specifiction {self.kernel}.")
        return None


    def fit(self, X, y, sample_weight=None):
        print(X, y)
        X, y = check_X_y(X, y, dtype=[np.float64, np.float32], force_all_finite=False)

        max_iter = 1000 if self.max_iter == -1 else self.max_iter
        self._c_svm = PyClassificationSvm(self.kernel, self.C, self.tol,
                                          max_iter, self.cache_size, True)
        self._c_svm.train(X, y, sample_weight)
        print(self._c_svm.get_support_indices())

        # cdef descriptor[float] desc
        # desc.set_c(self.C)
        # desc.set_accuracy_threshold(self.tol)

                # cdef descriptor[float, thunder, classification] desc(kernel_desc)


#         # self._set_base_attributes(output_type=X, target_dtype=y, n_features=X)

        # print('x_table: ')
        # cdef table * x_table = <table *>_input_to_onedal_table(<PyObject*>X)

        # print('y_table: ')
        # cdef table y_table
        # y_table = <table>_input_to_onedal_table(<PyObject*>y)
#         sample_weight_table = _input_to_onedal_table(sample_weight)

#         self.dtype = get_type(X)

#         cdef kernel_desc _kernel_desc = self._get_kernel_params(x_table)
#         cdef svm_descriptor _svm_desc = self._get_svm_params(_kernel_desc)

#         # cdef train_result_svm[classification] train_result = train(svm_descriptor, x_train, y_train)




#         # if self.dtype == np.float32:
#         #     train(svm_descriptor, x_train, y_train)
#         #     self._model = <uintptr_t>model_f
#         # elif self.dtype == np.float64:
#         #     model_d = new svmModel[double]()
#         #     train(svm_descriptor, x_train, y_train)
#         #     self._model = <uintptr_t>model_d
#         # else:
#         #     raise TypeError('Input data type should be float32 or float64')

#         self._unpack_model()
#         self._fit_status_ = 0

#         del x_table
#         del y_table
        print('End SVC FIT')
        return self

    def predict(self, X):

        return np.array()

