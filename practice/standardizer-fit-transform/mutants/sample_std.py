import numpy as np


def fit_standardizer(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=1) if len(x) > 1 else np.zeros(x.shape[1])
    scale = np.where(std > 0, std, 1.0)
    return mean, scale


def apply_standardizer(x, mean, scale):
    return (x - mean) / scale
