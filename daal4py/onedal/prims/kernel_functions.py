from sklearn.utils.validation import check_X_y, check_array
import numpy as np


def linear_kernel(X, Y, scale=1.0, shiht=0.0):
    # TODO
    from _onedal4py_host import PyLinearKernelParams, PyLinearKernelCompute

    X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
    Y = check_array(Y, dtype=[np.float64, np.float32], force_all_finite=False)

    self._onedal_params = PyLinearKernelParams(scale=scale, shiht=shiht)
    c_kernel = PyLinearKernelCompute(self._onedal_params)
    c_kernel.compute(X, Y)
    return c_svm.get_values()

def rbf_kernel(X, Y, sigma=1.0):
    # TODO
    from _onedal4py_host import PyLinearKernelParams, PyLinearKernelCompute

    X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
    Y = check_array(Y, dtype=[np.float64, np.float32], force_all_finite=False)

    self._onedal_params = PyRbfKernelParams(sigma=sigma)
    c_kernel = PyRbfKernelCompute(self._onedal_params)
    c_kernel.compute(X, Y)
    return c_kernel.get_values()

def poly_kernel(X, Y, scale=1.0, shiht=0.0, degree=3):
    # TODO
    from _onedal4py_host import PyLinearKernelParams, PyLinearKernelCompute

    X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
    Y = check_array(Y, dtype=[np.float64, np.float32], force_all_finite=False)

    self._onedal_params = PyLinearKernelParams(scale=scale, shiht=shiht, degree=degree)
    c_kernel = PyLinearKernelCompute(self._onedal_params)
    c_kernel.compute(X, Y)
    return c_kernel.get_values()

