A computer can make uniform random numbers between 0 and 1. All other random numbers come from transforming them. The functions here take the uniform numbers as input, so the tests are exact and do not depend on a random generator. The same idea runs inside every sampler, including the token sampler of a language model.

## Task

Write three functions. They all work on arrays and give the sample for each input value.

1. `exponential_from_uniform(u, rate)` uses the inverse CDF. The exponential CDF is $F(x) = 1 - e^{-\lambda x}$, so the sample for $u$ is $x = -\ln(1 - u) / \lambda$. For $u$ near 0, use `np.log1p` so that small values keep their precision.
2. `box_muller(u1, u2)` returns two arrays of standard normal samples `(z0, z1)`:

$$r = \sqrt{-2 \ln u_1}, \qquad z_0 = r \cos(2\pi u_2), \qquad z_1 = r \sin(2\pi u_2)$$

3. `categorical_from_uniform(u, probs)` returns the category index for each $u$. Build the cumulative sum of `probs`. The category of $u$ is the first index whose cumulative value is **greater than** $u$. A $u$ that equals a cumulative value belongs to the next category. Rounding must never give an index outside the list: cap the index at `len(probs) - 1`.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `u`, `u1`, `u2` | float array | any shape | uniform numbers. `u1` is greater than 0 |
| `rate` | float | scalar | the rate $\lambda$, greater than 0 |
| `probs` | float array | `(K,)` | the probability of each category, adding up to 1 |

- `exponential_from_uniform` returns a float array with the shape of `u`.
- `box_muller` returns two float arrays with the shape of `u1`.
- `categorical_from_uniform` returns an int array with the shape of `u`.

## Constraints

- Do not use `np.random` or `np.random.Generator`. The random numbers are inputs.
- Do not change the inputs.
- Use numpy only.

## Examples

```python
exponential_from_uniform(np.array([0.0, 0.5, 0.9]), 2.0)
```

```text
array([0.        , 0.34657359, 1.15129255])
```

```python
z0, z1 = box_muller(np.array([0.5]), np.array([0.0]))
float(z0[0])
float(z1[0])
```

```text
1.1774100225154747
0.0
```

```python
categorical_from_uniform(np.array([0.0, 0.3, 0.5, 0.99]), np.array([0.2, 0.3, 0.5]))
```

```text
array([0, 1, 2, 2])
```
