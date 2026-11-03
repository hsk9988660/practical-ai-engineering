The singular value decomposition writes a matrix as $A = U \Sigma V^\top$. The singular values in $\Sigma$ tell you how much each direction contributes to $A$. If you keep only the largest ones, you get the closest matrix of a lower rank. This is the idea behind image compression, recommender systems and LoRA.

## Task

Write three functions. You can use `np.linalg.svd`.

1. `low_rank(a, k)` returns the best rank-`k` approximation of `a`: $U_k \Sigma_k V_k^\top$, which keeps the `k` largest singular values. With `k = 0` it returns a matrix of zeros.
2. `energy_ratio(a, k)` returns the share of the energy that the first `k` singular values keep:

$$\frac{\sum_{i \le k} \sigma_i^2}{\sum_i \sigma_i^2}$$

If the matrix is all zeros, return `1.0`.

3. `smallest_rank(a, energy)` returns the smallest `k` for which `energy_ratio(a, k) >= energy`. For the zero matrix, return `0`.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `a` | float array | `(m, n)` | the matrix |
| `k` | int | scalar | the number of singular values to keep, from 0 to `min(m, n)` |
| `energy` | float | scalar | the target share of the energy, greater than 0 and at most 1 |

- `low_rank` returns a float array with the shape `(m, n)`.
- `energy_ratio` returns a float. `smallest_rank` returns an int.

## Constraints

- Do not change `a`.
- Use numpy only.

## Examples

```python
a = np.array([[3.0, 0.0], [0.0, 1.0]])
np.round(low_rank(a, 1), 6) + 0.0
energy_ratio(a, 1)
smallest_rank(a, 0.85)
smallest_rank(a, 0.95)
```

```text
array([[3., 0.],
       [0., 0.]])
0.9
1
2
```
