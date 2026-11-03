## Hint 1

Write one small function that checks a distribution and returns it as a float array. Use it in all three functions.

## Hint 2

To change the base, divide a natural log by `np.log(base)`. For base 2 this gives bits.

## Hint 3

Do not compute `p * np.log(p)` for the whole array: `0 * log(0)` is `0 * -inf`, which is NaN. Build a mask `p > 0` and sum only over the entries where it is true.

## Hint 4

For the cross-entropy, use the same mask for $p$. If any $q_i$ is 0 where the mask is true, return `float("inf")` before you take the log. The KL divergence is then the cross-entropy minus the entropy, and it is `inf` in the same case.
