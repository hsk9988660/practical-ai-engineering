import math


class Value:
    def __init__(self, data, children=()):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __pow__(self, n):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def relu(self):
        raise NotImplementedError

    def exp(self):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
