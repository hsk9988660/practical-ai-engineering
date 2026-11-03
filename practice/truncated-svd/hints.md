## Hint 1

`np.linalg.svd(a, full_matrices=False)` returns `u`, `s` and `vt`. The singular values `s` are sorted from large to small.

## Hint 2

To keep `k` values, use the first `k` columns of `u`, the first `k` values of `s` and the first `k` rows of `vt`. `u[:, :k] * s[:k]` scales each column of `u` by its singular value.

## Hint 3

The energy of a singular value is its square. Use `np.cumsum(s ** 2) / np.sum(s ** 2)` to get the energy ratio for every `k` at once.

## Hint 4

For `smallest_rank`, find the first index where the cumulative ratio reaches the target. `np.nonzero(...)[0][0] + 1` gives the rank, because an index of 0 means rank 1.
