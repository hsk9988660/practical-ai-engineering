import numpy as np


def rosenbrock(p):
    x, y = p
    return float((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)


def rosenbrock_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])


def gradient_descent(start, lr, steps):
    p = np.array(start, dtype=float)
    history = [rosenbrock(p)]
    for _ in range(steps):
        p = p + lr * rosenbrock_grad(p)
        history.append(rosenbrock(p))
    return p, history
