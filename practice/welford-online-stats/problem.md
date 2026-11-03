Data often arrives as a stream, or it is too large for memory, or it sits on many machines. You still need its mean and variance, for example to normalize features or to monitor a model. Welford's method keeps three numbers and updates them one value at a time. Unlike the textbook formula, it does not lose accuracy when the values are large and close together.

## Task

The state is a tuple `(n, mean, m2)`: the number of values so far, their mean, and `m2`, the sum of squared distances to the mean. The empty state is `(0, 0.0, 0.0)`.

Write three functions.

1. `welford_update(state, x)` adds one value and returns the new state:

$$n \leftarrow n + 1, \qquad \delta = x - \text{mean}, \qquad \text{mean} \leftarrow \text{mean} + \delta / n, \qquad m_2 \leftarrow m_2 + \delta\,(x - \text{mean}_{\text{new}})$$

2. `welford_merge(a, b)` returns the state of the union of two data parts (Chan's formula). With $n = n_a + n_b$ and $\delta = \text{mean}_b - \text{mean}_a$:

$$\text{mean} = \text{mean}_a + \delta\,\frac{n_b}{n}, \qquad m_2 = m_{2a} + m_{2b} + \delta^2\,\frac{n_a n_b}{n}$$

If both states are empty, return the empty state.

3. `welford_finalize(state, ddof=0)` returns `(mean, variance)` with `variance = m2 / (n - ddof)`. Raise `ValueError` if `n - ddof` is not positive.

## Input and output

| Name | Type | Meaning |
|---|---|---|
| `state`, `a`, `b` | tuple `(int, float, float)` | `(n, mean, m2)` |
| `x` | float | one new value |
| `ddof` | int | 0 for the population variance, 1 for the sample variance |

- `welford_update` and `welford_merge` return a new state tuple.
- `welford_finalize` returns a tuple of two floats.

## Constraints

- Do not keep the values. The state has three numbers only.
- Do not use the formula mean of squares minus square of mean.
- Use the standard library only.

## Examples

```python
from functools import reduce
data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
state = reduce(welford_update, data, (0, 0.0, 0.0))
state
welford_finalize(state)
welford_finalize(state, ddof=1)
welford_merge(reduce(welford_update, data[:4], (0, 0.0, 0.0)), reduce(welford_update, data[4:], (0, 0.0, 0.0)))
```

```text
(8, 5.0, 32.0)
(5.0, 4.0)
(5.0, 4.571428571428571)
(8, 5.0, 32.0)
```
