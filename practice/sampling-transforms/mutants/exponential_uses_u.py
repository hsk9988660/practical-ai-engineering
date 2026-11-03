import numpy as np


def exponential_from_uniform(u, rate):
    return -np.log(np.asarray(u, dtype=float)) / rate


def box_muller(u1, u2):
    radius = np.sqrt(-2.0 * np.log(np.asarray(u1, dtype=float)))
    angle = 2.0 * np.pi * np.asarray(u2, dtype=float)
    return radius * np.cos(angle), radius * np.sin(angle)


def categorical_from_uniform(u, probs):
    cdf = np.cumsum(np.asarray(probs, dtype=float))
    index = np.searchsorted(cdf, np.asarray(u, dtype=float), side="right")
    return np.minimum(index, len(cdf) - 1)
