## Why pivot

Without a swap, a zero on the diagonal stops the method, although the matrix can be invertible, as in the matrix that swaps two coordinates. A small non-zero pivot is a quieter problem. You divide by it, and the rounding errors in the other entries grow by the same factor. Choosing the largest entry in the column keeps all multipliers at most 1 in absolute value. This is partial pivoting, and it is what LU routines in libraries do.

## What singular means here

A matrix is singular when its rows depend on each other. Then at some column all candidate pivots are zero, up to rounding. The tolerance `1e-12` decides what counts as zero. A matrix that is almost singular passes the check but gives an inverse with huge entries. The condition number tells you how much to trust it.

## Should you invert at all

In practice you rarely compute $A^{-1}$. To solve $Ax = b$, an LU or QR solve is faster and more accurate than computing the inverse and multiplying. The inverse shows up in formulas, for example in the normal equation of linear regression, where the code should still use a solve.

## Cost

The method does about $n^3$ operations. The loop over the rows in the solution is simple but not fast. Libraries use blocked and vectorized routines.

## Where this goes next

The linear systems lesson solves $Ax = b$ directly. The linear regression lesson uses the normal equation, and the Gaussian and covariance material uses inverses and determinants.
