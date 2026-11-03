## The problem with the textbook formula

The variance is $E[x^2] - E[x]^2$ on paper. For values near $10^9$ with a spread of 1, both terms are near $10^{18}$ and their difference is near 1. A float64 has about 16 digits, so the result has no correct digit, and it can even be negative. Welford's method only works with distances to the running mean, which are small.

## Why merge matters

Many systems compute statistics in parallel: each worker folds its part, and the states are merged. Batch normalization in distributed training, data validation pipelines and monitoring systems all do this. The merge is associative and commutative, so the order does not matter.

## Population and sample variance

With `ddof=0` you divide by $n$. With `ddof=1` you divide by $n - 1$, which makes the estimate of the variance unbiased. One value has no sample variance, so the function raises an error and does not return NaN.

## Where this goes next

Batch normalization keeps a running mean and variance during training, with a moving average. The standardizer exercise uses a fixed mean and scale. Both are special cases of the ideas here.
