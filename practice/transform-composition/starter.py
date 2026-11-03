import numpy as np


def rotation(theta):
    raise NotImplementedError


def scaling(sx, sy):
    raise NotImplementedError


def shear(kx, ky):
    raise NotImplementedError


def compose(*matrices):
    raise NotImplementedError


def apply(matrix, points):
    raise NotImplementedError
