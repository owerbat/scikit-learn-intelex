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
from numpy.testing import assert_allclose
from daal4py.onedal.prims import linear_kernel

SCALES = [1.0, 2.0]
SHIFTS = [0.0, 1.0]
ARRAY_CONSTRS = [np.array]
DTYPES = [np.float32, np.float64]

def _test_dense_small_linear_kernel(scale, shift, array_constr, dtype):
    rng = np.random.RandomState(0)
    X = array_constr(5 * rng.random_sample((5, 4)), dtype=dtype)
    Y = array_constr(5 * rng.random_sample((3, 4)), dtype=dtype)

    result = linear_kernel(X, Y, scale=scale, shift=shift)
    expected = np.dot(X, np.array(Y).T) * scale + shift
    assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize('array_constr', ARRAY_CONSTRS)
@pytest.mark.parametrize('scale', SCALES)
@pytest.mark.parametrize('shift', SHIFTS)
@pytest.mark.parametrize('dtype', DTYPES)
def test_dense_small_linear_kernel(scale, shift, array_constr, dtype):
    _test_dense_small_linear_kernel(scale, shift, array_constr, dtype)
