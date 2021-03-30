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

from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.deprecation import deprecated

from sklearn.base import RegressorMixin as sklearn_RegressorMixin
from sklearn.svm import SVR as sklearn_SVR

from ._base import BaseLibSVM


class SVR(sklearn_RegressorMixin, BaseLibSVM):

    _impl = 'epsilon_svr'

    @_deprecate_positional_args
    def __init__(self, *, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, nu=0., epsilon=epsilon, verbose=verbose,
            shrinking=shrinking, probability=False, cache_size=cache_size,
            class_weight=None, max_iter=max_iter, random_state=None)

    @deprecated(
        "The probA_ attribute is deprecated in version 0.23 and will be "
        "removed in version 1.0 (renaming of 0.25).")
    @property
    def probA_(self):
        return self._probA

    @deprecated(
        "The probB_ attribute is deprecated in version 0.23 and will be "
        "removed in version 1.0 (renaming of 0.25).")
    @property
    def probB_(self):
        return self._probB

    def _more_tags(self):
        return sklearn_SVR._more_tags(self)
