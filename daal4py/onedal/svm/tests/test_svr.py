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
from sklearn import datasets, metrics
from sklearn.metrics.pairwise import rbf_kernel

from daal4py.onedal.svm import SVR
from sklearn.svm import SVR as SklearnSVR

def test_diabetes_simple():
    diabetes = datasets.load_diabetes()
    clf = SVR(kernel='linear', C=10.)
    clf.fit(diabetes.data, diabetes.target)
    assert clf.score(diabetes.data, diabetes.target) > 0.02

def test_predict():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    reg = SVR(kernel='linear', C=0.1).fit(X, y)

    linear = np.dot(X, reg.support_vectors_.T)
    dec = np.dot(linear, reg.dual_coef_.T) + reg.intercept_
    assert_array_almost_equal(dec.ravel(), reg.predict(X).ravel())

    reg = SVR(kernel='rbf', gamma=1).fit(X, y)

    rbfs = rbf_kernel(X, reg.support_vectors_, gamma=reg.gamma)
    dec = np.dot(rbfs, reg.dual_coef_.T) + reg.intercept_
    assert_array_almost_equal(dec.ravel(), reg.predict(X).ravel())

def _test_fit_compare_with_sklearn(kernel):
    diabetes = datasets.load_diabetes()
    clf = SVR(kernel=kernel, C=10.)
    clf.fit(diabetes.data, diabetes.target)

    clf_sk = SklearnSVR(kernel=kernel, C=10.)
    clf_sk.fit(diabetes.data, diabetes.target)

    # assert_array_almost_equal(clf.dual_coef_, clf_sk.dual_coef_)
    assert_array_almost_equal(clf.support_, clf_sk.support_)
    assert_array_almost_equal(clf.support_vectors_, clf_sk.support_vectors_)
    assert_array_almost_equal(clf.intercept_, clf_sk.intercept_)


@pytest.mark.parametrize('kernel', ['linear', 'rbf', 'poly'])
def test_fit_compare_with_sklearn(kernel):
    _test_fit_compare_with_sklearn(kernel)


