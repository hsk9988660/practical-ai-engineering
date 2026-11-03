The Gaussian (normal) distribution describes many measurements and the noise in many models. A model that predicts a mean and a spread is trained by maximizing the likelihood of the data. Training code works with the log of the density, because the density itself can be too small for a float.

## Task

Write three functions.

1. `gaussian_log_pdf(x, mu, sigma)` returns the log of the normal density, for one number or an array of numbers:

$$\log p(x) = -\tfrac{1}{2}\log\!\left(2\pi\sigma^2\right) - \frac{(x-\mu)^2}{2\sigma^2}$$

Raise `ValueError` if `sigma` is not positive.

2. `fit_gaussian(x)` returns `(mu, sigma)`, the maximum likelihood estimate for the data `x`. The estimate of $\mu$ is the mean. The estimate of $\sigma$ is the standard deviation with division by $n$.

3. `log_likelihood(x, mu, sigma)` returns the sum of `gaussian_log_pdf` over all numbers in `x`, as a float.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `x` | float or float array | scalar or `(n,)` | the data |
| `mu` | float | scalar | the mean |
| `sigma` | float | scalar | the standard deviation, greater than 0 |

- `gaussian_log_pdf` returns a float or a float array with the shape of `x`.
- `fit_gaussian` returns a tuple of two floats.
- `log_likelihood` returns a float.

## Constraints

- Compute the log-density directly from the formula. Do not take the log of the density.
- Do not change `x`.
- Use numpy only.

## Examples

```python
gaussian_log_pdf(0.0, 0.0, 1.0)
mu, sigma = fit_gaussian(np.array([1.0, 2.0, 3.0, 4.0]))
mu, sigma
log_likelihood(np.array([1.0, 2.0, 3.0, 4.0]), mu, sigma)
```

```text
-0.9189385332046727
(2.5, 1.118033988749895)
-6.12204123544711
```
