import numpy as np


def gaussian_log_pdf(x, mu, sigma):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    x = np.asarray(x, dtype=float)
    return -np.log(2 * np.pi * sigma ** 2) - (x - mu) ** 2 / (2 * sigma ** 2)


def fit_gaussian(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x))


def log_likelihood(x, mu, sigma):
    return float(np.sum(gaussian_log_pdf(x, mu, sigma)))
