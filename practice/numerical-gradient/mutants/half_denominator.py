import numpy as np


def numerical_gradient(f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step.flat[i] = h
        grad.flat[i] = (f(x + step) - f(x - step)) / h
    return grad


def relative_error(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) + np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)
