import numpy as np


def check_distribution(p):
    p = np.asarray(p, dtype=float)
    if np.any(p < 0) or abs(p.sum() - 1.0) > 1e-8:
        raise ValueError("p must be a probability distribution")
    return p


def entropy(p, base=2):
    p = check_distribution(p)
    nz = p >= 0
    return float(-np.sum(p[nz] * np.log(p[nz])) / np.log(base)) + 0.0


def cross_entropy(p, q, base=2):
    p = check_distribution(p)
    q = check_distribution(q)
    nz = p > 0
    if np.any(q[nz] == 0):
        return float("inf")
    return float(-np.sum(p[nz] * np.log(q[nz])) / np.log(base)) + 0.0


def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)
