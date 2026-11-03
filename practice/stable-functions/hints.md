## Hint 1

For `logsumexp`, subtract the maximum along the axis before you call `exp`. Then the largest exponent is 0 and nothing overflows. Add the maximum back after the log.

## Hint 2

A maximum of $-\infty$ makes `x - max` equal to NaN. Replace a non-finite maximum by 0 before you subtract.

## Hint 3

For `softplus`, use the identity $\log(1 + e^x) = \max(x, 0) + \log(1 + e^{-|x|})$. The argument of `exp` is never positive. Use `np.log1p`, which keeps the precision of $\log(1 + y)$ for a small $y$.

## Hint 4

For `sigmoid`, use $1 / (1 + e^{-x})$ for $x \ge 0$ and $e^{x} / (1 + e^{x})$ for $x < 0$. In both cases the argument of `exp` is not positive. Pick the branch with a boolean mask.

## Hint 5

`log_softmax` is `x` minus `logsumexp` with the reduced axis put back. `np.expand_dims` does this.
