An orthonormal basis is a set of vectors with length 1 that are at right angles to each other. Such a basis makes many computations simple: coordinates are dot products, and projections need no matrix inverse. The Gram-Schmidt method builds one from any list of vectors.

## Task

Write `gram_schmidt(vectors, tol=1e-10)`.

Go through the rows of `vectors` in order. For each row $v$:

1. Subtract from $v$ its projection on every basis vector $q$ that you already have: $w \leftarrow w - (q \cdot w)\, q$. Start with $w = v$ and use the updated $w$ in each step.
2. If the length of $w$ is greater than `tol`, divide $w$ by its length and add it to the basis. If not, the row depends on the earlier rows, so skip it.

Return the basis as an array with one row for each basis vector, in the order in which you found them.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `vectors` | float array | `(n, d)` | one vector in each row |
| `tol` | float | scalar | a residual of this length or less counts as zero |

- The function returns a float array with the shape `(k, d)`. Here `k` is the number of independent rows, so `k <= min(n, d)`. If `k` is 0, the shape is `(0, d)`.

## Constraints

- Do not change `vectors`.
- Do not call `np.linalg.qr` or other library routines that do the orthogonalization for you.
- Use numpy only.

## Examples

```python
gram_schmidt(np.array([[3.0, 1.0], [2.0, 2.0]]))
gram_schmidt(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))
```

```text
array([[ 0.9486833 ,  0.31622777],
       [-0.31622777,  0.9486833 ]])
array([[1.        , 0.        , 0.        ],
       [0.        , 0.70710678, 0.70710678]])
```
