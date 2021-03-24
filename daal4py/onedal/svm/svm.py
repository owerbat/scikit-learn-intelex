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

from sklearn.base import ClassifierMixin

import numpy as np
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


class SVC(ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', *, gamma=1.0,
                 coef0=0.0, tol=1e-3, cache_size=200.0, max_iter=-1,
                 class_weight=None,  decision_function_shape='ovr',
                 break_ties=False, algorithm='thunder', **kwargs):

        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.algorithm = algorithm

    @_execute_with_dpc_or_host("PyClassificationSvmTrain", "PyClassificationSvmParams")
    def fit(self, X, y, sample_weight=None):
        # todo
        PyClassificationSvmTrain = getattr(import_module('_onedal4py_host'), 'PyClassificationSvmTrain')
        PyClassificationSvmParams = getattr(import_module('_onedal4py_host'), 'PyClassificationSvmParams')

        X, y = _check_X_y(X, y, dtype=[np.float64, np.float32], force_all_finite=False)
        y, self.class_weight_, self.classes_ = _validate_targets(y, self.class_weight, X.dtype)
        print(type(X), type(y))
        print(X.dtype, y.dtype)
        sample_weight = _get_sample_weight(X, y, sample_weight, self.class_weight_, self.classes_)
        print(sample_weight.shape, X.shape, y.shape)
        print(sample_weight)
        max_iter = 1000 if self.max_iter == -1 else self.max_iter
        self._onedal_params = PyClassificationSvmParams(self.algorithm, self.kernel)
        self._onedal_params.c = self.C
        self._onedal_params.class_count = len(self.classes_)
        self._onedal_params.accuracy_threshold = self.tol
        self._onedal_params.max_iteration_count = self.max_iter

        c_svm = PyClassificationSvmTrain(self._onedal_params)
        c_svm.train(X, y, sample_weight)

        self.dual_coef_ = c_svm.get_coeffs().T
        self.support_vectors_ = c_svm.get_support_vectors()
        self.intercept_ = c_svm.get_biases().ravel()
        self.support_ = c_svm.get_support_indices().ravel()

        print('End SVC FIT')
        return self

    def predict(self, X):
        if self.break_ties and self.decision_function_shape == 'ovo':
            raise ValueError("break_ties must be False when "
                             "decision_function_shape is 'ovo'")

        if self.break_ties and self.decision_function_shape == 'ovr' and \
                len(self.classes_) > 2:
            y = np.argmax(self.decision_function(X), axis=1)
        else:
            from _onedal4py_host import PyClassificationSvmInfer
            X = _check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
            print('Infer: start')
            c_svm = PyClassificationSvmInfer(self._onedal_params)
            c_svm.infer(X, self.support_vectors_, self.dual_coef_.T, self.intercept_)
            print('Infer: finish')
            y = c_svm.get_labels()
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
