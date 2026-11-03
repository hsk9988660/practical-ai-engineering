The inverse of a matrix $A$ is the matrix $A^{-1}$ with $A A^{-1} = I$. You find it with row operations: write $A$ next to the identity matrix, change the rows until the left half is the identity, and read the inverse from the right half. This method is called Gauss-Jordan elimination.

## Task

Write `inverse(a)`.

1. Make the augmented matrix $[A \mid I]$ with $n$ rows and $2n$ columns.
2. For each column `col` from 0 to $n - 1$:
   1. Pivot: from the rows `col` to $n - 1$, find the row where the absolute value of the entry in this column is largest. Swap it with row `col`.
   2. If this largest absolute value is less than `1e-12`, the matrix is singular. Raise `ValueError`.
   3. Divide row `col` by the pivot, so that the pivot becomes 1.
   4. For every other row, subtract that row's entry in this column times row `col`. The column then has a 1 on the diagonal and 0 elsewhere.
3. Return the right half.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `a` | float array | `(n, n)` | a square matrix |

- The function returns a float array with the shape `(n, n)`.
- The function raises `ValueError` when the matrix is singular.

## Constraints

- Do not change `a`.
- Do not call `np.linalg.inv`, `np.linalg.solve` or similar functions.
- Use numpy only.

## Examples

```python
inverse(np.array([[2.0, 1.0], [5.0, 3.0]]))
inverse(np.array([[0.0, 1.0], [1.0, 0.0]]))
inverse(np.array([[1.0, 2.0], [2.0, 4.0]]))
```

```text
array([[ 3., -1.],
       [-5.,  2.]])
array([[0., 1.],
       [1., 0.]])
ValueError: matrix is singular
```
