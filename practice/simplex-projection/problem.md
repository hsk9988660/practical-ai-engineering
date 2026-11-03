Many problems have constraints: the weights of a mixture must be non-negative and add up to 1, for example. Projected gradient descent handles this in two steps: take a normal gradient step, and then move the result to the nearest point that satisfies the constraint. For the set of probability vectors (the simplex), this nearest point has a fast exact formula.

## Task

Write two functions.

1. `project_simplex(v, z=1.0)` returns the point of the set $\{x : x_i \ge 0,\ \sum_i x_i = z\}$ that is nearest to `v` in the Euclidean distance. The answer has the form $x_i = \max(v_i - \theta, 0)$. Find $\theta$ like this:
   1. Sort `v` in descending order to get `u`.
   2. Compute the cumulative sums `css = cumsum(u) - z`.
   3. Let `rho` be the largest index $j$ (counting from 1) for which $u_j - \text{css}_j / j > 0$.
   4. Then $\theta = \text{css}_{\rho} / \rho$.
2. `projected_gradient(grad, x0, project, lr, steps)` first projects `x0`. Then it repeats `steps` times: $x \leftarrow \text{project}(x - \text{lr}\cdot \text{grad}(x))$. It returns the final `x`.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `v` | float array | `(n,)` | any real vector |
| `z` | float | scalar | the sum of the simplex, greater than 0 |
| `grad` | function | | takes `x` and returns the gradient, an array with the shape of `x` |
| `x0` | float array | `(n,)` | the start point |
| `project` | function | | maps a vector to its nearest feasible point, for example `project_simplex` |
| `lr` | float | scalar | the learning rate |
| `steps` | int | scalar | the number of iterations, at least 0 |

- Both functions return a float array with the shape `(n,)`.

## Constraints

- Do not change `v` or `x0`.
- Use numpy only.

## Examples

```python
project_simplex(np.array([2.0, 0.0]))
project_simplex(np.array([1.0, 1.0, 1.0]))
project_simplex(np.array([0.8, 0.1, -0.4]))
```

```text
array([1., 0.])
array([0.33333333, 0.33333333, 0.33333333])
array([0.85, 0.15, 0.  ])
```

```python
c = np.array([2.0, 0.0, 0.0])
np.round(projected_gradient(lambda x: 2 * (x - c), np.full(3, 1 / 3), project_simplex, 0.25, 50), 6) + 0.0
```

```text
array([1., 0., 0.])
```
