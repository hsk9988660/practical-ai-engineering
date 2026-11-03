import numpy as np


def to_polar(z):
    z = np.asarray(z, dtype=complex)
    return np.hypot(z.real, z.imag), np.arctan2(z.imag, z.real)


def from_polar(r, theta):
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return r * np.cos(theta) + 1j * r * np.sin(theta)


def complex_power(z, n):
    r, theta = to_polar(z)
    return from_polar(r ** n, n * theta)


def roots_of_unity(n):
    return from_polar(1.0, 2.0 * np.pi * np.arange(n) / n)


def nth_roots(z, n):
    r, theta = to_polar(z)
    k = np.arange(n)
    return from_polar(r ** (1.0 / n), theta / n + 2.0 * np.pi * k)
