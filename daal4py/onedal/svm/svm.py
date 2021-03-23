from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import numpy as np
from importlib import import_module
from daal4py.onedal.common import _execute_with_dpc_or_host

# 2 optional
# try:
#     import dpctl
#     print('DPCTL: _onedal4py_dpc')
#     from _onedal4py_dpc import PyClassificationSvm
#     # from _onedal4py_host cimport *
# except ImportError:
#     print('HOST: _onedal4py_host')
#     from _onedal4py_host import PyClassificationSvm

# 3 optional

# win and linux only for python37
# from _onedal4py_dpc import PyClassificationSvm

# mac and other python
# from _onedal4py_host import PyClassificationSvm


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

    @_execute_with_dpc_or_host("PyClassificationSvmTrain", "PyClassificationSvmParams")
    def fit(self, X, y, sample_weight=None):
        # todo
        PyClassificationSvmTrain = getattr(import_module('_onedal4py_host'), 'PyClassificationSvmTrain')
        PyClassificationSvmParams = getattr(import_module('_onedal4py_host'), 'PyClassificationSvmParams')

        X, y = check_X_y(X, y, dtype=[np.float64, np.float32], force_all_finite=False)

        max_iter = 1000 if self.max_iter == -1 else self.max_iter
        self._onedal_params = PyClassificationSvmParams(self.algorithm, self.kernel)
        self._onedal_params.c = self.C
        self._onedal_params.accuracy_threshold = self.tol

        c_svm = PyClassificationSvmTrain(self._onedal_params)
        c_svm.train(X, y, sample_weight)
        print("get_support_indices: ", c_svm.get_support_indices())
        print("get_support_vectors: ", c_svm.get_support_vectors())
        print("get_bias", c_svm.get_biases())
        print("get_coeffs", c_svm.get_coeffs())
        self.dual_coef_ = c_svm.get_coeffs()
        self.support_vectors_ = c_svm.get_support_vectors()
        self.intercept_ = c_svm.get_biases()

        print('End SVC FIT')
        return self

    def predict(self, X):
        from _onedal4py_host import PyClassificationSvmInfer
        X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
        print('Infer: start')
        c_svm = PyClassificationSvmInfer(self._onedal_params)
        c_svm.infer(X, self.support_vectors_, self.dual_coef_, self.intercept_)
        print('Infer: finish')
        return c_svm.get_labels()


    def decision_function(self, X):
        from _onedal4py_host import PyClassificationSvmInfer

        X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
        print('decision_function: start')
        c_svm = PyClassificationSvmInfer(self._onedal_params)
        c_svm.infer(X, self.support_vectors_, self.dual_coef_, self.intercept_)
        print('decision_function: finish')
        return c_svm.get_decision_function()
