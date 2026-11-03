import numpy as np


def numerical_gradient(f, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        x.flat[i] += h
        plus = f(x)
        x.flat[i] -= 2 * h
        minus = f(x)
        grad.flat[i] = (plus - minus) / (2 * h)
    return grad


def relative_error(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) + np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)
