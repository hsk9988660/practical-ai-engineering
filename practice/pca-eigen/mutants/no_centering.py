import numpy as np


def fit_pca(x, k):
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0)
    centered = x - mean
    cov = x.T @ x / (len(x) - 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    vectors = vectors[:, order]
    for j in range(vectors.shape[1]):
        if vectors[np.argmax(np.abs(vectors[:, j])), j] < 0:
            vectors[:, j] = -vectors[:, j]
    ratio = values / values.sum()
    return mean, vectors[:, :k].T, ratio[:k]


def transform(x, mean, components):
    return (np.asarray(x, dtype=float) - mean) @ components.T


def inverse_transform(z, mean, components):
    return np.asarray(z, dtype=float) @ components + mean
