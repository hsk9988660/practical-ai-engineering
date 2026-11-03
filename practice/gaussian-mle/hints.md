## Hint 1

Write the formula as two terms. The first term does not depend on `x`. The second term is the squared distance to the mean, divided by $2\sigma^2$.

## Hint 2

Do not call `np.exp` anywhere in `gaussian_log_pdf`. At a point 40 standard deviations from the mean the density is about $10^{-348}$, which a float64 shows as 0, and the log of 0 is minus infinity.

## Hint 3

`np.std` divides by $n$ by default. This is the maximum likelihood estimate. Convert the results with `float(...)`.

## Hint 4

`log_likelihood` is one line: sum the log-densities. The log of a product of independent densities is the sum of their logs.
