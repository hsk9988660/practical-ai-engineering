import numpy as np


def project_simplex(v, z=1.0):
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - z
    index = np.arange(1, len(v) + 1)
    rho = index[u - css / index > 0][0]
    theta = css[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def projected_gradient(grad, x0, project, lr, steps):
    x = project(np.asarray(x0, dtype=float))
    for _ in range(steps):
        x = project(x - lr * grad(x))
    return x
