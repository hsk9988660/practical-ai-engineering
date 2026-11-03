import numpy as np


def check_stochastic(p):
    p = np.asarray(p, dtype=float)
    if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("each row must be a probability distribution")
    return p


def n_step(p0, p, n):
    p = check_stochastic(p)
    dist = np.asarray(p0, dtype=float)
    for _ in range(n + 1):
        dist = dist @ p
    return dist


def stationary_distribution(p):
    p = check_stochastic(p)
    k = p.shape[0]
    system = p.T - np.eye(k)
    system[-1] = 1.0
    rhs = np.zeros(k)
    rhs[-1] = 1.0
    return np.linalg.solve(system, rhs)


def simulate(p, start, uniforms):
    p = check_stochastic(p)
    cdf = np.cumsum(p, axis=1)
    state = int(start)
    path = [state]
    for u in uniforms:
        state = int(min(np.searchsorted(cdf[state], u, side="right"), len(cdf) - 1))
        path.append(state)
    return path
