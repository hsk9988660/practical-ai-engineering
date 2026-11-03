## Hint 1

For the exponential, solve $u = 1 - e^{-\lambda x}$ for $x$. Use `np.log1p(-u)`, which is $\ln(1 - u)$.

## Hint 2

For Box-Muller, compute the radius and the angle once. Return `radius * cos(angle)` and `radius * sin(angle)`.

## Hint 3

`np.searchsorted(cdf, u, side="right")` gives the first index whose cumulative value is greater than `u`. This is the category.

## Hint 4

The sum of `probs` can be a little below 1 because of rounding. For `u` very near 1, `searchsorted` can then return `K`, which is not a valid index. Cap the result with `np.minimum`.
