When you write a backward pass by hand, a small mistake gives a model that trains badly and does not tell you why. A gradient check compares your analytic gradient with a numerical estimate. If the two agree, the backward pass is very likely right.

## Task

Write two functions.

1. `numerical_gradient(f, x, h=1e-6)` estimates the gradient of the scalar function `f` at the array `x`. For every entry $i$ of `x`, use the central difference:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h\, e_i) - f(x - h\, e_i)}{2h}$$

where $e_i$ is the array that has a 1 at entry $i$ and 0 elsewhere. The function returns an array with the same shape as `x`.

2. `relative_error(a, b)` compares two arrays. It returns

$$\frac{\lVert a - b \rVert}{\lVert a \rVert + \lVert b \rVert}$$

with the 2-norm over all entries. If both arrays are zero, it returns `0.0`.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `f` | function | | takes a float array of the shape of `x` and returns a float |
| `x` | float array | any shape | the point where you estimate the gradient |
| `h` | float | scalar | the size of the step |
| `a`, `b` | float arrays | same shape | the two gradients to compare |

- `numerical_gradient` returns a float array with the shape of `x`.
- `relative_error` returns a float.

## Constraints

- Do not change `x`. Call `f` with new arrays.
- `x` can have any number of dimensions, for example a weight matrix.
- Use numpy only.

## Examples

```python
f = lambda v: v[0] ** 2 + 3 * v[0] * v[1]
np.round(numerical_gradient(f, np.array([1.0, 2.0])), 4)
round(relative_error(np.array([1.0, 0.0]), np.array([1.0, 0.001])), 6)
relative_error(np.zeros(3), np.zeros(3))
```

```text
array([8., 3.])
0.0005
0.0
```
