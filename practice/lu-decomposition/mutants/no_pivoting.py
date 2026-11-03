import numpy as np


def lu_decompose(a):
    u = np.array(a, dtype=float)
    n = u.shape[0]
    perm = np.arange(n)
    l = np.zeros((n, n))
    for k in range(n):
        pivot = k
        if abs(u[pivot, k]) < 1e-12:
            raise ValueError("matrix is singular")
        if pivot != k:
            u[[k, pivot]] = u[[pivot, k]]
            l[[k, pivot], :k] = l[[pivot, k], :k]
            perm[[k, pivot]] = perm[[pivot, k]]
        for i in range(k + 1, n):
            l[i, k] = u[i, k] / u[k, k]
            u[i, k:] -= l[i, k] * u[k, k:]
    return perm, l + np.eye(n), np.triu(u)


def forward_substitution(l, b):
    b = np.asarray(b, dtype=float)
    y = np.zeros(len(b))
    for i in range(len(b)):
        y[i] = (b[i] - l[i, :i] @ y[:i]) / l[i, i]
    return y


def back_substitution(u, b):
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - u[i, i + 1:] @ x[i + 1:]) / u[i, i]
    return x


def lu_solve(a, b):
    perm, l, u = lu_decompose(a)
    y = forward_substitution(l, np.asarray(b, dtype=float)[perm])
    return back_substitution(u, y)
