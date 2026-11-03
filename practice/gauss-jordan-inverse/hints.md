## Hint 1

Build the augmented matrix with `np.hstack([a, np.eye(n)])`. This also makes a copy, so you do not change `a`.

## Hint 2

Row swaps are easy in numpy: `work[[i, j]] = work[[j, i]]`.

## Hint 3

To find the pivot row, look only at the rows from `col` down: `col + np.argmax(np.abs(work[col:, col]))`. If you use the diagonal entry, a zero on the diagonal breaks the method, and a very small entry loses precision.

## Hint 4

After you scale the pivot row, loop over all other rows, above and below. Subtract `work[row, col] * work[col]` from each. Check the pivot size before you divide.
