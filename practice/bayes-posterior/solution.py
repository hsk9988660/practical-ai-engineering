import numpy as np


def posterior(prior, likelihood):
    joint = np.asarray(prior, dtype=float) * np.asarray(likelihood, dtype=float)
    total = joint.sum()
    if total == 0:
        raise ValueError("the evidence is zero")
    return joint / total


def sequential_posterior(prior, likelihoods):
    current = np.asarray(prior, dtype=float)
    for row in np.asarray(likelihoods, dtype=float):
        current = posterior(current, row)
    return current
