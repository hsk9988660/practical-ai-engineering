import numpy as np


def dft(x):
    x = np.asarray(x, dtype=complex)
    n = len(x)
    k = np.arange(n)
    matrix = np.exp(2j * np.pi * np.outer(k, k) / n)
    return matrix @ x


def fft(x):
    x = np.asarray(x, dtype=complex)
    n = len(x)
    if n == 0 or n & (n - 1):
        raise ValueError("the length must be a power of 2")
    if n == 1:
        return x.copy()
    even = fft(x[0::2])
    odd = fft(x[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate([even + twiddle, even - twiddle])


def ifft(spectrum):
    spectrum = np.asarray(spectrum, dtype=complex)
    return np.conj(fft(np.conj(spectrum))) / len(spectrum)
