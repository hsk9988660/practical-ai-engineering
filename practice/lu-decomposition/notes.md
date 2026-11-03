## Why factor

Solving a triangular system costs $n^2$ operations. The factorization costs about $n^3 / 3$, and you pay it once. If you have many right-hand sides for the same matrix, you reuse $L$ and $U$ and only pay $n^2$ for each. Computing the inverse costs about three times more than the factorization and is less accurate.

## Pivoting

Without row swaps, the method fails on a zero pivot and loses accuracy on a small one, as in the Gauss-Jordan exercise. Choosing the largest entry in the column keeps every multiplier at most 1, so the rounding errors do not grow fast. The permutation is part of the answer: $P A = L U$.

## The hidden detail

When two rows swap at step $k$, the multipliers that you stored in the earlier columns belong to those rows and must swap with them. This is the most common bug in a hand-written LU, and it only appears for matrices of size 3 and above.

## Singular matrices

If all candidates for the pivot are zero, the matrix is singular and the system has no unique solution. The code raises an error and does not divide by zero.

## Where this goes next

Linear regression with the normal equation solves a linear system. Gaussian processes and many second-order optimizers solve systems with a symmetric matrix, where the Cholesky factorization is a faster special case of the same idea.
