## Hint 1

Broadcasting lines up shapes from the right. A `(C,)` array meets the last axis of `x`, which is `W`, and not the channel axis. Add axes with `None` to move the `C` numbers to axis 1.

## Hint 2

For `one_hot`, make an array of zeros and set the ones with one indexing step: `out[np.arange(N), labels] = 1.0`. Check the range of the labels first, because a negative label would silently index from the end.

## Hint 3

For `batched_outer`, give `a` a new last axis and `b` a new middle axis: shapes `(N, d1, 1)` and `(N, 1, d2)`. Their product broadcasts to `(N, d1, d2)`.

## Hint 4

For `masked_mean`, give the mask a last axis so that it broadcasts over `D`. Multiply by the mask, sum over time, and divide by the number of valid steps. Use `np.maximum(count, 1)` in the division, and then replace the rows that have no valid step by 0.
