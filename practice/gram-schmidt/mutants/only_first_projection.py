import numpy as np


def gram_schmidt(vectors, tol=1e-10):
    vectors = np.asarray(vectors, dtype=float)
    basis = []
    for v in vectors:
        w = v.copy()
        for q in basis[:1]:
            w = w - np.dot(q, w) * q
        norm = np.linalg.norm(w)
        if norm > tol:
            basis.append(w / norm)
    return np.array(basis).reshape(len(basis), vectors.shape[1])
