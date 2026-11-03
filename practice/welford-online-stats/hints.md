## Hint 1

In `welford_update`, compute `delta` with the old mean. Update the mean. Then multiply `delta` by the distance to the **new** mean for `m2`. The two different means are what makes the update exact.

## Hint 2

`m2` is not the variance. It is the variance times `n` (for ddof 0). Keep it as a sum so that the update stays simple.

## Hint 3

In `welford_merge`, the term `delta ** 2 * na * nb / n` accounts for the difference between the two means. Check the empty case first, to avoid a division by zero.

## Hint 4

To test your code, fold the data in one pass, and also fold two halves and merge them. Both must give the same state.
