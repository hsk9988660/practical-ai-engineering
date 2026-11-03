## Hint 1

Write a helper that checks the matrix first and returns it as a float array. Use it in all three functions.

## Hint 2

A distribution is a row vector, so one step is `dist @ p`. The order matters: `p @ dist` multiplies the wrong way.

## Hint 3

For the stationary distribution, the system $(P^\top - I)\pi = 0$ alone is singular, because $\pi = 0$ solves it. Replacing the last equation by $\sum \pi_i = 1$ makes the solution unique.

## Hint 4

For `simulate`, build the cumulative sums of every row once with `np.cumsum(p, axis=1)`. Then use `np.searchsorted(cdf[state], u, side="right")` to pick the next state. Cap it at `k - 1` in case of rounding.
