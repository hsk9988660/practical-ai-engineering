import numpy as np


def posterior(prior, likelihood):
    joint = np.asarray(prior, dtype=float) * np.asarray(likelihood, dtype=float)
    total = joint.sum()
    if total == 0:
        raise ValueError("the evidence is zero")
    return joint / total


def sequential_posterior(prior, likelihoods):
    likelihoods = np.asarray(likelihoods, dtype=float)
    return posterior(prior, np.prod(likelihoods, axis=0) if len(likelihoods) else np.ones(len(prior)))
