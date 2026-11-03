import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    out[~positive] = e / (1.0 + e)
    return out


def softplus(x):
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def logsumexp(x, axis=-1):
    x = np.asarray(x, dtype=float)
    axis = -1
    top = np.max(x, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):
        total = np.log(np.sum(np.exp(x - top), axis=axis, keepdims=True))
    return np.squeeze(top + total, axis=axis)


def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    return x - np.expand_dims(logsumexp(x, axis), axis)
