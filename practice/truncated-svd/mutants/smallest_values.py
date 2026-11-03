import numpy as np


def low_rank(a, k):
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return (u[:, -k:] * s[-k:]) @ vt[-k:]


def energy_ratio(a, k):
    s = np.linalg.svd(a, compute_uv=False)
    total = np.sum(s ** 2)
    if total == 0:
        return 1.0
    return float(np.sum(s[:k] ** 2) / total)


def smallest_rank(a, energy):
    s = np.linalg.svd(a, compute_uv=False)
    total = np.sum(s ** 2)
    if total == 0:
        return 0
    cumulative = np.cumsum(s ** 2) / total
    reached = np.nonzero(cumulative >= energy)[0]
    return int(reached[0]) + 1
