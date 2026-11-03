import numpy as np


def inverse(a):
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    work = np.hstack([a, np.eye(n)])
    for col in range(n):
        rows = np.nonzero(np.abs(work[col:, col]) > 1e-12)[0]
        if len(rows) == 0:
            raise ValueError("matrix is singular")
        pivot = col + int(rows[0])
        work[[col, pivot]] = work[[pivot, col]]
        work[col] = work[col] / work[col, col]
        for row in range(n):
            if row != col:
                work[row] = work[row] - work[row, col] * work[col]
    return work[:, n:]
