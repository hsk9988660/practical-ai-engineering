import numpy as np


def inverse(a):
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    work = np.hstack([a, np.eye(n)])
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(work[col:, col])))
        work[[col, pivot]] = work[[pivot, col]]
        with np.errstate(divide="ignore", invalid="ignore"):
            work[col] = work[col] / work[col, col]
        for row in range(n):
            if row != col:
                work[row] = work[row] - work[row, col] * work[col]
    return work[:, n:]
