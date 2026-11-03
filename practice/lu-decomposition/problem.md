To solve $A x = b$, you do not compute the inverse of $A$. You factor $A$ once into a lower and an upper triangular matrix, and then solve two easy triangular systems. This is what `np.linalg.solve` does inside. You write the whole chain.

## Task

Write four functions.

1. `lu_decompose(a)` returns `(perm, l, u)` with `a[perm] = l @ u`.
   - Work on a copy of `a`. For each column `k` from 0 to $n - 1$, find the row `p >= k` with the largest absolute value in column `k`. If that value is below `1e-12`, raise `ValueError`.
   - If `p != k`, swap rows `k` and `p` in the working matrix, in the part of `l` that you already filled (columns before `k`), and in `perm`. `perm` starts as `[0, 1, ..., n-1]`.
   - For every row `i > k`, the multiplier is `m = u[i, k] / u[k, k]`. Store it in `l[i, k]` and subtract `m` times row `k` from row `i`.
   - `l` has ones on the diagonal and `u` is upper triangular.
2. `forward_substitution(l, b)` solves $L y = b$ for a lower triangular `l`, from the first row down.
3. `back_substitution(u, b)` solves $U x = y$ for an upper triangular `u`, from the last row up.
4. `lu_solve(a, b)` solves $A x = b$: factor, apply the permutation to `b` (`b[perm]`), then forward and back substitution.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `a` | float array | `(n, n)` | a square matrix |
| `b` | float array | `(n,)` | the right-hand side |
| `l`, `u` | float arrays | `(n, n)` | a lower or an upper triangular matrix with a non-zero diagonal |

- `lu_decompose` returns an int array of shape `(n,)` and two float arrays of shape `(n, n)`.
- The other three functions return a float array of shape `(n,)`.

## Constraints

- Do not change `a` or `b`.
- Do not call `np.linalg.solve`, `np.linalg.inv` or a library LU routine.
- Use numpy only.

## Examples

```python
a = np.array([[2.0, 1.0], [4.0, 3.0]])
factors = lu_decompose(a)
factors[0]
factors[1]
factors[2]
lu_solve(a, np.array([3.0, 7.0]))
```

```text
array([1, 0])
array([[1. , 0. ],
       [0.5, 1. ]])
array([[ 4. ,  3. ],
       [ 0. , -0.5]])
array([1., 1.])
```
