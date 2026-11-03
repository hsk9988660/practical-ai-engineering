A matrix can move, turn, stretch or shear every point of the plane. When you apply several transformations one after the other, you can multiply their matrices into one matrix. The order of the product is the part that people get wrong.

## Task

Write five functions. All matrices are 2 by 2. A point is a row of 2 numbers, and the matrix acts on the point as a column vector: $p' = M p$.

1. `rotation(theta)` returns the matrix that turns points by the angle `theta` in radians, counter-clockwise: $\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$.
2. `scaling(sx, sy)` returns the diagonal matrix with `sx` and `sy`.
3. `shear(kx, ky)` returns $\begin{pmatrix}1 & k_x\\ k_y & 1\end{pmatrix}$.
4. `compose(*matrices)` returns one matrix that has the same effect as applying the matrices one after the other, **first argument first**. With no arguments it returns the identity matrix.
5. `apply(matrix, points)` returns the points after the transformation.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `theta`, `sx`, `sy`, `kx`, `ky` | float | scalar | the parameters of a transformation |
| `matrices` | float arrays | `(2, 2)` each | the transformations, in the order in which they are applied |
| `matrix` | float array | `(2, 2)` | one transformation |
| `points` | float array | `(n, 2)` | one point in each row |

- `rotation`, `scaling`, `shear` and `compose` return a float array with the shape `(2, 2)`.
- `apply` returns a float array with the shape `(n, 2)`. Row $i$ is the matrix times point $i$.

## Constraints

- Do not change the inputs.
- Use numpy only.

## Examples

```python
compose(scaling(2.0, 1.0), shear(1.0, 0.0))
compose(shear(1.0, 0.0), scaling(2.0, 1.0))
apply(compose(scaling(2.0, 1.0), shear(1.0, 0.0)), np.array([[1.0, 0.0], [0.0, 1.0]]))
compose()
```

```text
array([[2., 1.],
       [0., 1.]])
array([[2., 2.],
       [0., 1.]])
array([[2., 0.],
       [1., 1.]])
array([[1., 0.],
       [0., 1.]])
```
