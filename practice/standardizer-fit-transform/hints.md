## Hint 1

A column is one feature, so the statistics run down the rows. In numpy this is `axis=0`.

## Hint 2

`np.std` divides by $n$ by default. This is the population standard deviation that the task asks for.

## Hint 3

A constant column has a standard deviation of 0. Replace each 0 by 1.0 with `np.where`. Then the division in `apply_standardizer` gives 0 for that column and not NaN.

## Hint 4

`apply_standardizer` is one expression. Numpy broadcasts the `(n_features,)` arrays over the rows of `x`. Write `(x - mean) / scale`, which makes a new array.
