import numpy as np


def channel_normalize(x, mean, std):
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)[None, :, None, None]
    std = np.asarray(std, dtype=float)[None, :, None, None]
    return (x - mean) / std


def one_hot(labels, n):
    labels = np.asarray(labels, dtype=int)
    out = np.zeros((len(labels), n))
    out[np.arange(len(labels)), labels] = 1.0
    return out


def batched_outer(a, b):
    return np.asarray(a, dtype=float)[:, :, None] * np.asarray(b, dtype=float)[:, None, :]


def masked_mean(x, mask):
    x = np.asarray(x, dtype=float)
    keep = np.asarray(mask, dtype=bool)[:, :, None]
    total = (x * keep).sum(axis=1)
    count = keep.sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)
