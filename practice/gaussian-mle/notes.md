## Why the log

The likelihood of many points is a product of many densities. Each factor is small, so the product soon becomes 0 in floating point. The log turns the product into a sum, which stays in a safe range. Working with the log-density from the start also avoids the case where one far-away point has a density that rounds to 0.

## Why the estimate divides by n

The likelihood is largest when $\mu$ is the sample mean and $\sigma^2$ is the average squared distance to it. This is the division by $n$. The estimate with $n - 1$ is unbiased for the variance, but it is not the maximum of the likelihood. For large $n$ the two are almost equal.

## What the tests check

- At the mean, the log-density has only the first term.
- No other pair of parameters gives a larger log-likelihood than the fitted pair.
- A point very far from the mean gives a large negative number and not minus infinity.

## Where this goes next

Many losses are a negative log-likelihood. Mean squared error is the negative log-likelihood of a Gaussian with a fixed $\sigma$. Cross-entropy is the negative log-likelihood of a categorical distribution. Variational autoencoders and diffusion models use Gaussian log-densities in their objectives.
