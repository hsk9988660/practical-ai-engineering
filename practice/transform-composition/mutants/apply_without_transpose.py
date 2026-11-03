import numpy as np


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def scaling(sx, sy):
    return np.array([[sx, 0.0], [0.0, sy]])


def shear(kx, ky):
    return np.array([[1.0, kx], [ky, 1.0]])


def compose(*matrices):
    result = np.eye(2)
    for m in matrices:
        result = np.asarray(m, dtype=float) @ result
    return result


def apply(matrix, points):
    return np.asarray(points, dtype=float) @ np.asarray(matrix, dtype=float)
