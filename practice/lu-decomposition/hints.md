## Hint 1

Start with `u = np.array(a, dtype=float)`, which is a copy. At step `k`, the pivot row is `k + np.argmax(np.abs(u[k:, k]))`.

## Hint 2

A row swap must also swap the multipliers that you already stored. Swap only the columns before `k` of `l`, because the later columns are still empty. If you forget this, the factorization is wrong for a matrix of size 3 or more.

## Hint 3

Return `l + np.eye(n)` so that the diagonal of `l` is ones. Return `np.triu(u)` to remove the tiny leftovers below the diagonal.

## Hint 4

Forward substitution: `y[i] = (b[i] - l[i, :i] @ y[:i]) / l[i, i]`, for `i` from 0 up. Back substitution has the same form from the last row up, and uses `u[i, i+1:] @ x[i+1:]`.

## Hint 5

In `lu_solve`, apply the permutation to the right-hand side. The factorization is of `a[perm]`, so the system is `l u x = b[perm]`.
