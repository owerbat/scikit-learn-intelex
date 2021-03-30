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

from abc import ABCMeta, abstractmethod
from sklearn.base import ClassifierMixin as sklearn_ClassifierMixin
from sklearn.svm._base import BaseLibSVM as sklearn_BaseLibSVM
from sklearn.svm._base import BaseSVC as sklearn_BaseSVC


class BaseLibSVM(sklearn_BaseLibSVM):
    @abstractmethod
    def __init__(self, kernel, degree, gamma, coef0,
                 tol, C, nu, epsilon, shrinking, probability, cache_size,
                 class_weight, verbose, max_iter, random_state):
        super().__init__(
            self, kernel, degree, gamma, coef0,
            tol, C, nu, epsilon, shrinking, probability, cache_size,
            class_weight, verbose, max_iter, random_state)

    def fit(self, X, y, sample_weight=None):
        sklearn_BaseLibSVM.fit(self, X, y, sample_weight)

    def predict(self, X):
        sklearn_BaseLibSVM.predict(self, X)

class BaseSVC(sklearn_ClassifierMixin, BaseLibSVM, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, kernel, degree, gamma, coef0, tol, C, nu,
                 shrinking, probability, cache_size, class_weight, verbose,
                 max_iter, decision_function_shape, random_state,
                 break_ties):
        sklearn_BaseSVC.__init__(
            self, kernel, degree, gamma, coef0, tol, C, nu,
            shrinking, probability, cache_size, class_weight, verbose,
            max_iter, decision_function_shape, random_state,
            break_ties)

    def decision_function(self, X):
        return sklearn_BaseSVC.decision_function(self, X)

    def predict(self, X):
        return sklearn_BaseSVC.predict(self, X)

    @property
    def predict_proba(self):
        return sklearn_BaseSVC.predict_proba

    @property
    def predict_log_proba(self):
        return sklearn_BaseSVC.predict_log_proba

    @property
    def probA_(self):
        return sklearn_BaseSVC._probA

    @property
    def probB_(self):
        return sklearn_BaseSVC._probB
