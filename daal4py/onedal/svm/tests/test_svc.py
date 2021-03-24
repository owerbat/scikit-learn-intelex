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

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from daal4py.onedal.svm import SVC

def _test_libsvm_parameters(array_constr, dtype):
    X = array_constr([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype=dtype)
    y = array_constr([1, 1, 1, 2, 2, 2], dtype=dtype)

    clf = SVC(kernel='linear').fit(X, y)
    assert_array_equal(clf.dual_coef_, [[-0.25, .25]])
    assert_array_equal(clf.support_, [1, 3])
    assert_array_equal(clf.support_vectors_, (X[1], X[3]))
    assert_array_equal(clf.intercept_, [0.])
    assert_array_equal(clf.predict(X), y)

@pytest.mark.parametrize('array_constr', [np.array])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_libsvm_parameters(array_constr, dtype):
    _test_libsvm_parameters(array_constr, dtype)

def test_class_weight():
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = SVC(class_weight={1: 0.1})
    clf.fit(X, y)
    assert_array_almost_equal(clf.predict(X), [2] * 6)

def test_sample_weight():
    X = np.array([[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = SVC(kernel='linear')
    clf.fit(X, y, sample_weight=[1] * 6)
    assert_array_almost_equal(clf.intercept_, [0.0])


