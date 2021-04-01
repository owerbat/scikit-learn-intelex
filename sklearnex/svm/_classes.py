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

import numpy as np
from scipy import sparse as sp
# import logging

from sklearn.svm import SVR as sklearn_SVR
from sklearn.svm import SVC as sklearn_SVC
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.multiclass import _ovr_decision_function
from sklearn.preprocessing import LabelEncoder

from onedal.svm import SVR as onedal_SVR
from onedal.svm import SVC as onedal_SVC
from onedal.common.validation import _column_or_1d


class SVR(sklearn_SVR):
    @_deprecate_positional_args
    def __init__(self, *, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C,
            epsilon=epsilon, shrinking=shrinking, cache_size=cache_size, verbose=verbose,
            max_iter=max_iter)

    def fit(self, X, y, sample_weight=None):
        if self.kernel in ['linear', 'rbf', 'poly'] and not sp.isspmatrix(X):
            print('BaseLibSVM.fit oneDAL')
            # logging.info("sklearn.svm.SVR.fit: " + get_patch_message("daal"))

            self._onedal_model = onedal_SVR(C=self.C, epsilon=self.epsilon,
                                            kernel=self.kernel, degree=self.degree,
                                            gamma=self.gamma, coef0=self.coef0,
                                            tol=self.tol, shrinking=self.shrinking,
                                            cache_size=self.cache_size,
                                            max_iter=self.max_iter)
            self._onedal_model.fit(X, y, sample_weight)

            self.support_vectors_ = self._onedal_model.support_vectors_
            self.n_features_in_ = self._onedal_model.n_features_in_
            self.fit_status_ = 0
            self.dual_coef_ = self._onedal_model.dual_coef_
            self.intercept_ = self._onedal_model.intercept_
            self.shape_fit_ = self._onedal_model.shape_fit_
            self.support_ = self._onedal_model.support_

            self._dual_coef_ = self.dual_coef_
            self._n_support = [self.support_vectors_.shape[0],]
            self._sparse = False
            self._probA = None
            self._probB = None
        else:
            print('BaseLibSVM.fit stock')
            # logging.info("sklearn.svm.SVR.fit: " + get_patch_message("sklearn"))
            sklearn_SVR.fit(self, X, y, sample_weight)

        return self

    def predict(self, X):
        if hasattr(self, '_onedal_model') and not sp.isspmatrix(X):
            print('BaseLibSVM.predict oneDAL')
            return self._onedal_model.predict(X)
        else:
            print('BaseLibSVM.predict stock')
            return sklearn_SVR.predict(self, X)


class SVC(sklearn_SVC):
    @_deprecate_positional_args
    def __init__(self, *, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 break_ties=False, random_state=None):
        super().__init__(
            C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape, break_ties=break_ties,
            random_state=random_state)

    def _compute_balanced_class_weight(self, y):
        y_ = _column_or_1d(y)
        classes, _ = np.unique(y_, return_inverse=True)

        le = LabelEncoder()
        y_ind = le.fit_transform(y_)
        if not all(np.in1d(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        recip_freq = len(y_) / (len(le.classes_) *
                               np.bincount(y_ind).astype(np.float64))
        return recip_freq[le.transform(classes)]

    def fit(self, X, y, sample_weight=None):
        if self.kernel in ['linear', 'rbf', 'poly'] and not sp.isspmatrix(X):
            print('sklearn.svm.SVC.fit oneDAL')
            # logging.info("sklearn.svm.SVC.fit: " + get_patch_message("daal"))

            self._onedal_model = onedal_SVC(
                C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                coef0=self.coef0, tol=self.tol, shrinking=self.shrinking,
                cache_size=self.cache_size, max_iter=self.max_iter,
                class_weight=self.class_weight, break_ties=self.break_ties,
                decision_function_shape=self.decision_function_shape)
            self._onedal_model.fit(X, y, sample_weight)

            if self.class_weight == 'balanced':
                self.class_weight_ = self._compute_balanced_class_weight(y)
            else:
                self.class_weight_ = self._onedal_model.class_weight_

            self.support_vectors_ = self._onedal_model.class_weight_
            self.fit_status_ = 0
            self.dual_coef_ = self._onedal_model.dual_coef_
            self.intercept_ = self._onedal_model.intercept_
            self.shape_fit_ = self._onedal_model.class_weight_
            self.classes_ = self._onedal_model.classes_
            self.support_ = self._onedal_model.support_

            self._n_support = self._onedal_model._n_support
            self._sparse = False
        else:
            print('sklearn.svm.SVC.fit stock')
            # logging.info("sklearn.svm.SVC.fit: " + get_patch_message("sklearn"))
            sklearn_SVC.fit(self, X, y, sample_weight)

        return self

    def predict(self, X):
        if hasattr(self, '_onedal_model') and not sp.isspmatrix(X):
            print('sklearn.svm.SVC.predict oneDAL')
            return self._onedal_model.predict(X)
        else:
            print('sklearn.svm.SVC.predict stock')
            return sklearn_SVC.predict(self, X)

    def decision_function(self, X):
        if hasattr(self, '_onedal_model') and not sp.isspmatrix(X):
            print('sklearn.svm.SVC.decision_function oneDAL')
            dec = self._onedal_model.decision_function(X)
        else:
            print('sklearn.svm.SVC.decision_function stock')
            dec = sklearn_SVC.decision_function(self, X)

        if self.decision_function_shape == 'ovr' and len(self.classes_) > 2:
            return _ovr_decision_function(dec < 0, -dec, len(self.classes_))

        return dec
