"""
Data utils for SVMp+
Author: Fan Feng
"""
import numpy as np


def get_kernel(x, kernel='linear', **kwargs):
    """
    Parameters
    ------------
        x: numpy.array
            input data (s * d), s entries of data, d dimensions
        kernel: str
            kernel type, "linear" or "gaussian"
        sigma: float
            if using gaussian kernel, the sigma must be given

    Returns
    ------------
        K: numpy.array
            kernel matrix (s * s)
    """
    if kernel == 'linear':
        K = x.dot(x.T)
    elif kernel == 'gaussian':
        sigma = kwargs.get('sigma', 1)
        K = np.ones((len(x), len(x)))
        for i in range(len(x)-1):
            for j in range(i+1, len(x)):
                diff = x[i] - x[j]
                val = np.exp(- np.inner(diff, diff) / 2 / sigma**2)
                K[i, j] = val
                K[j, i] = val
    else:
        raise ValueError('Unsupported kernel!')
    return K


def get_Hessian(y, x_kernel, xp_kernel, gamma):
    """
    Parameters
    ------------
        y: numpy.array
            training labels (n-dim)
        x_kernel: numpy.array
            kernel matrix of training data x (n * n)
        xp_kernel: numpy.array
            kernel matrix of training privileged data xp (m * m)
        gamma: float
            gamma in the objective function

    Returns
    ------------
        H: numpy.array
            kernel matrix ((m+n) * (m+n))
        """
    n, m = len(x_kernel), len(xp_kernel)
    H = np.zeros((n+m, n+m))
    H_mini = - xp_kernel / gamma
    # print(H_mini)
    if m != 0:
        H[:m, :m] = H_mini
        H[n:, :m] = H_mini
        H[:m, n:] = H_mini
        H[n:, n:] = H_mini
    H_mini_2 = - np.outer(y, y) * x_kernel
    # print(H_mini_2)
    H[:n, :n] += H_mini_2
    return H


def get_obj_func_value(p, y, x_kernel, xp_kernel, gamma, Cs):
    """
    Parameters
    ------------
        p: numpy.array
            current point ((n+m)-dim)
        y: numpy.array
            training labels (n-dim)
        x_kernel: numpy.array
            kernel matrix of training data x (n * n)
        xp_kernel: numpy.array
            kernel matrix of training privileged data xp (m * m)
        gamma: float
            gamma in the objective function
        Cs: float
            C^(star), the weight for privileged information

    Returns
    ------------
        val: float
            objective function value
        """
    n, m = len(x_kernel), len(xp_kernel)
    a1, a2 = p[:n], p[:m]
    b1 = p[n:]
    a1y = a1 * y
    a2cs = a2 + b1 - Cs
    if m != 0:
        val = np.sum(a1) - a1y.dot(x_kernel).dot(a1y) / 2 - a2cs.dot(xp_kernel).dot(a2cs) / 2 / gamma
    else:
        val = np.sum(a1) - y.dot(x_kernel).dot(y) / 2
    return val


def get_gradient_at_point(p, y, x_kernel, xp_kernel, gamma, Cs):
    """
    Parameters
    ------------
        p: numpy.array
            current point ((n+m)-dim)
        y: numpy.array
            training labels (n-dim)
        x_kernel: numpy.array
            kernel matrix of training data x (n * n)
        xp_kernel: numpy.array
            kernel matrix of training privileged data xp (m * m)
        gamma: float
            gamma in the objective function
        Cs: float
            C^(star), the weight for privileged information

    Returns
    ------------
        grad: numpy.array
            gradient at current point ((n+m)-dim)
        """
    n, m = len(x_kernel), len(xp_kernel)
    a1, a2 = p[:n], p[:m]
    b1 = p[n:]
    a1y = a1 * y
    a2cs = a2 + b1 - Cs
    grad = np.zeros(p.shape)
    if m != 0:
        c1 = - a2cs.dot(xp_kernel) / gamma
        grad[:m] = c1
        grad[n:] = c1
    grad[:n] += 1
    grad[:n] -= a1y.dot(x_kernel) * y
    return grad


def generate_start(n, m, Cs):
    """
    Generate the start point
        """
    p = np.zeros((n+m,))
    p[n:] = Cs
    return p


def normalize_features(features):
    """
    Normalize the n * d features by each column
    """
    _sum = np.max(np.abs(features), axis=1)
    _sum[_sum == 0] = 1
    _features = np.zeros(features.shape)
    for i in range(features.shape[0]):
        _features[i, :] = features[i, :] / _sum[i]
    return _features


def normalize_features_0(features):
    """
    Normalize the n * d features by each column
    """
    _sum = np.max(features, axis=0)
    _sum[_sum == 0] = 1
    _features = np.zeros(features.shape)
    for i in range(features.shape[1]):
        _features[:, i] = features[:, i] / _sum[i]
    return _features

