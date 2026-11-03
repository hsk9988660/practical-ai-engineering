import numpy as np


def fit_standardizer(x):
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    return mean, scale


def apply_standardizer(x, mean, scale):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - mean) / scale
