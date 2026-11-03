import numpy as np


def inverse(a):
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    work = np.hstack([a, np.eye(n)])
    for col in range(n):
        pivot = col + int(np.argmax(np.abs(work[col:, col])))
        if abs(work[pivot, col]) < 1e-12:
            raise ValueError("matrix is singular")
        work[[col, pivot]] = work[[pivot, col]]
        work[col] = work[col] / work[col, col]
        for row in range(n):
            if row != col:
                work[row] = work[row] - work[row, col] * work[col]
    return work[:, n:]
