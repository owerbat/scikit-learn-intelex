from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import numpy as np

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


    def fit(self, X, y, sample_weight=None):
        print(X, y)


        # 1 optional

        # from _onedal4py_dpc import PyClassificationSvm
        # from _onedal4py_host import PyClassificationSvm

        if 'dpctl' in sys.modules:
            from dpctl import is_in_device_context
            if is_in_device_context():
                from _onedal4py_dpc import PyClassificationSvm
            else:
                from _onedal4py_host import PyClassificationSvm
        else:
            from _onedal4py_host import PyClassificationSvm
            print('HOST: _onedal4py_host')

        X, y = check_X_y(X, y, dtype=[np.float64, np.float32], force_all_finite=False)

        max_iter = 1000 if self.max_iter == -1 else self.max_iter
        self._c_svm = PyClassificationSvm(self.kernel, self.C, self.tol,
                                          max_iter, self.cache_size, True)
        self._c_svm.train(X, y, sample_weight)
        print("get_support_indices: ", self._c_svm.get_support_indices())
        print("get_support_vectors: ", self._c_svm.get_support_vectors())
        print("get_bias", self._c_svm.get_bias())
        print("get_coeffs", self._c_svm.get_coeffs())

        print('End SVC FIT')
        return self

    def predict(self, X):
        print('SVC PREDICT')
        # X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
        print('Infer: start')
        self._c_svm.infer(X)
        print('Infer: finish')
        print("get_labels: ", self._c_svm.get_labels())
        # print(self._c_svm.get_labels())
        return self._c_svm.get_labels()


    def decision_function(self, X):
        print('SVC PREDICT')
        # X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
        print('Infer: start')
        self._c_svm.infer(X)
        print('Infer: finish')
        print("get_decision_function: ", self._c_svm.get_decision_function())
        print("get_labels: ", self._c_svm.get_labels())
        # print(self._c_svm.get_labels())
        return self._c_svm.get_decision_function()

