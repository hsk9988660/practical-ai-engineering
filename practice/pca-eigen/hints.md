## Hint 1

Center the data first. The covariance matrix describes the spread around the mean, so it needs `x - mean`.

## Hint 2

`np.linalg.eigh` is for symmetric matrices. It returns the eigenvalues in ascending order, and the eigenvectors in the **columns** of the second result. `np.argsort(values)[::-1]` gives the order from large to small. Apply it to the values and to the columns.

## Hint 3

An eigenvector and its negative are both correct, and the library can return either. To make the result repeatable, loop over the columns. Find the entry with the largest absolute value with `np.argmax(np.abs(column))`, and multiply the column by -1 if that entry is negative.

## Hint 4

The ratio uses the sum of all eigenvalues, also those that you do not keep. Take the first `k` columns, transpose them so that components are rows, and slice `ratio[:k]`.
