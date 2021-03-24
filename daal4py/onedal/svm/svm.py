#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse as sp
from importlib import import_module
from daal4py.onedal.common import _execute_with_dpc_or_host
from daal4py.onedal.common import _validate_targets, _check_X_y, _check_array, _get_sample_weight

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

class BaseSVM(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, C, epsilon, kernel='rbf', *, degree, gamma,
                 coef0, tol, shrinking, cache_size, max_iter,
                 class_weight,  decision_function_shape,
                 break_ties, algorithm, **kwargs):

        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.tol = tol
        self.shrinking=shrinking
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.algorithm = algorithm


    def _compute_sigma(self, gamma, X):
        if gamma == 'scale':
            if sp.isspmatrix(X):
                # var = E[X^2] - E[X]^2
                X_sc = (X.multiply(X)).mean() - (X.mean())**2
            else:
                X_sc = X.var()
            _gamma = 1.0 / (X.shape[1] * X_sc) if X_sc != 0 else 1.0
        elif gamma == 'auto':
            _gamma = 1.0 / X.shape[1]
        else:
            _gamma = gamma
        return np.sqrt(0.5 / _gamma)


    def _fit(self, X, y, sample_weight, Computer):
        X, y = _check_X_y(X, y, dtype=[np.float64, np.float32], force_all_finite=False)
        y, self.class_weight_, self.classes_ = _validate_targets(y, self.class_weight, X.dtype)
        sample_weight = _get_sample_weight(X, y, sample_weight, self.class_weight_, self.classes_)

        PySvmParams = getattr(import_module('_onedal4py_host'), 'PySvmParams')

        self._onedal_params = PySvmParams(self.algorithm, self.kernel)
        self._onedal_params.c = self.C
        self._onedal_params.epsilon = self.epsilon
        self._onedal_params.class_count = len(self.classes_)
        self._onedal_params.accuracy_threshold = self.tol
        self._onedal_params.sigma = self._compute_sigma(self.gamma, X)
        self._onedal_params.max_iteration_count = 1000 if self.max_iter == -1 else self.max_iter

        c_svm = Computer(self._onedal_params)
        c_svm.train(X, y, sample_weight)

        self.dual_coef_ = c_svm.get_coeffs().T
        self.support_vectors_ = c_svm.get_support_vectors()
        self.intercept_ = c_svm.get_biases().ravel()
        self.support_ = c_svm.get_support_indices().ravel()
        return self

    def _predict(self, X, Computer):
        if self.break_ties and self.decision_function_shape == 'ovo':
            raise ValueError("break_ties must be False when "
                             "decision_function_shape is 'ovo'")

        if self.break_ties and self.decision_function_shape == 'ovr' and \
                len(self.classes_) > 2:
            y = np.argmax(self.decision_function(X), axis=1)
        else:
            X = _check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
            c_svm = Computer(self._onedal_params)
            c_svm.infer(X, self.support_vectors_, self.dual_coef_.T, self.intercept_)
            y = c_svm.get_labels()
        return y



class SVR(RegressorMixin, BaseSVM):
    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', *, degree=3,
                 gamma='scale', coef0=0.0, tol=1e-3, shrinking=True, cache_size=200.0,
                 max_iter=-1, algorithm='thunder', **kwargs):
        super().__init__(C=C, epsilon=epsilon, kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, shrinking=shrinking,
            cache_size=cache_size, max_iter=max_iter, class_weight=None,
            decision_function_shape=None, break_ties=False, algorithm=algorithm)

    @_execute_with_dpc_or_host("PyRegressionSvmTrain", "PySvmParams")
    def fit(self, X, y, sample_weight=None):
        return super()._fit(X, y, sample_weight, getattr(import_module('_onedal4py_host'), 'PyRegressionSvmTrain'))

    @_execute_with_dpc_or_host("PyRegressionSvmInfer")
    def predict(self, X):
        return super()._predict(X, getattr(import_module('_onedal4py_host'), 'PyRegressionSvmInfer'))


class SVC(ClassifierMixin, BaseSVM):
    def __init__(self, C=1.0, kernel='rbf', *, degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, shrinking=True, cache_size=200.0, max_iter=-1,
                 class_weight=None,  decision_function_shape='ovr',
                 break_ties=False, algorithm='thunder', **kwargs):
        super().__init__(C=C, epsilon=0.0, kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, shrinking=shrinking,
            cache_size=cache_size, max_iter=max_iter, class_weight=class_weight,
            decision_function_shape=decision_function_shape, break_ties=break_ties, algorithm=algorithm)

    @_execute_with_dpc_or_host("PyClassificationSvmTrain", "PySvmParams")
    def fit(self, X, y, sample_weight=None):
        return super()._fit(X, y, sample_weight, getattr(import_module('_onedal4py_host'), 'PyClassificationSvmTrain'))

    @_execute_with_dpc_or_host("PyClassificationSvmInfer", "PySvmParams")
    def predict(self, X):
        y = super()._predict(X, getattr(import_module('_onedal4py_host'), 'PyClassificationSvmInfer'))
        if len(self.classes_) == 2:
            y = y.ravel()
        return self.classes_.take(np.asarray(y, dtype=np.intp))


    def decision_function(self, X):
        from _onedal4py_host import PyClassificationSvmInfer

        X = _check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
        print('decision_function: start')
        c_svm = PyClassificationSvmInfer(self._onedal_params)
        c_svm.infer(X, self.support_vectors_, self.dual_coef_.T, self.intercept_)
        print('decision_function: finish')
        return c_svm.get_decision_function()
