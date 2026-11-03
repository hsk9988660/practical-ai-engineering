import numpy as np


def output_size(size, kernel, padding, stride):
    return (size + 2 * padding - kernel) // stride + 1


def conv2d(x, w, b=None, stride=1, padding=0):
    c_out, c_in, kh, kw = w.shape
    padded = np.pad(x, ((0, 0), (2 * padding, 0), (2 * padding, 0)))
    h_out = output_size(x.shape[1], kh, padding, stride)
    w_out = output_size(x.shape[2], kw, padding, stride)
    out = np.zeros((c_out, h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            patch = padded[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[:, i, j] = np.tensordot(w, patch, axes=([1, 2, 3], [0, 1, 2]))
    if b is not None:
        out = out + b[:, None, None]
    return out
