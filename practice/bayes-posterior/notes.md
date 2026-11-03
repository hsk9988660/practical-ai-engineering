## The base-rate effect

In the medical example a good test still gives a low posterior, because the disease is rare. Most positive results come from the large healthy group. People often forget the prior and read the likelihood as the answer. The code makes the role of the prior clear: without it, the posterior would be the normalized likelihood, which is 95 percent.

## The evidence is only a scale

The denominator does not depend on the hypothesis. It makes the posterior add up to 1. Because of this, you can multiply first and normalize later, or at each step, and get the same posterior. This is why the order of independent observations does not change the result.

## Why normalize at each step

A product of 1000 likelihoods of 0.1 is $10^{-1000}$, which is 0 in float64. The posterior would then be 0 divided by 0. Normalizing after each step keeps the numbers near 1. Many systems work in log space for the same reason, and then use the log-sum-exp trick to normalize.

## A hypothesis with prior 0 stays at 0

If a prior is exactly 0, no data can bring the belief back. This is why a prior should not be 0 for something that might be true.

## Where this goes next

The naive Bayes classifier multiplies the likelihoods of the features per class, in log space. Bayesian updates also appear in A/B tests, in filters such as the Kalman filter, and in the idea that a model's weights have a posterior.
