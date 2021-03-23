from sklearn.utils.validation import check_X_y, check_array
import numpy as np


def linear_kernel(X, Y, scale=1.0, shift=0.0):
    # TODO
    from _onedal4py_host import PyLinearKernelParams, PyLinearKernelCompute

    X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
    Y = check_array(Y, dtype=[np.float64, np.float32], force_all_finite=False)

    _onedal_params = PyLinearKernelParams(scale, shift)
    c_kernel = PyLinearKernelCompute(_onedal_params)
    c_kernel.compute(X, Y)
    return c_kernel.get_values()

def rbf_kernel(X, Y, sigma=1.0):
    # TODO
    from _onedal4py_host import PyRbfKernelParams, PyRbfKernelCompute

    X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
    Y = check_array(Y, dtype=[np.float64, np.float32], force_all_finite=False)

    _onedal_params = PyRbfKernelParams(sigma=sigma)
    c_kernel = PyRbfKernelCompute(_onedal_params)
    c_kernel.compute(X, Y)
    return c_kernel.get_values()

def poly_kernel(X, Y, scale=1.0, shift=0.0, degree=3):
    # TODO
    from _onedal4py_host import PyLinearKernelParams, PyLinearKernelCompute

    X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=False)
    Y = check_array(Y, dtype=[np.float64, np.float32], force_all_finite=False)

    _onedal_params = PyLinearKernelParams(scale=scale, shift=shift, degree=degree)
    c_kernel = PyLinearKernelCompute(_onedal_params)
    c_kernel.compute(X, Y)
    return c_kernel.get_values()

