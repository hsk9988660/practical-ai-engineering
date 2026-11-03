import numpy as np


def fit_standardizer(x):
    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1)
    scale = np.where(std > 0, std, 1.0)
    return mean, scale


def apply_standardizer(x, mean, scale):
    return (x - mean[:, None]) / scale[:, None]
