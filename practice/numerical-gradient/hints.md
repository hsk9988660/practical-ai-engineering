## Hint 1

Make a float copy of `x` and an array of zeros for the result, both with the same shape as `x`.

## Hint 2

You need to visit every entry, whatever the number of dimensions. The flat view `array.flat[i]` reads and writes entry `i` of an array of any shape.

## Hint 3

For each entry, make a `step` array of zeros with `h` at that entry. Evaluate `f(x + step)` and `f(x - step)`. These expressions make new arrays, so `x` stays as it is.

## Hint 4

For `relative_error`, compute the denominator first. If it is 0, both arrays are zero, so return `0.0`. Use `np.linalg.norm`, which flattens an array of any shape.
