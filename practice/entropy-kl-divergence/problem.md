Entropy measures how uncertain a distribution is. Cross-entropy is the average cost of coding data from $p$ with a code made for $q$, and it is the loss of most classifiers. The KL divergence is the extra cost, and it measures how far $q$ is from $p$.

## Task

Write three functions for discrete distributions. All use the log with the given `base`, which is 2 by default (bits).

1. `entropy(p, base=2)` returns $H(p) = -\sum_i p_i \log p_i$.
2. `cross_entropy(p, q, base=2)` returns $H(p, q) = -\sum_i p_i \log q_i$. If $q_i = 0$ for some $i$ with $p_i > 0$, the result is `float("inf")`.
3. `kl_divergence(p, q, base=2)` returns $D_{KL}(p \,\|\, q) = H(p, q) - H(p)$.

Terms with $p_i = 0$ count as 0, because $0 \log 0 = 0$ by convention. This is also true when $q_i = 0$.

`p` and `q` must be probability distributions: no negative entry, and a sum of 1 within `1e-8`. If not, raise `ValueError`.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `p`, `q` | float arrays | `(n,)` | discrete distributions over the same $n$ outcomes |
| `base` | float | scalar | the base of the logarithm: 2 for bits, `np.e` for nats |

- Every function returns a float. Entropy is never negative, and a result of zero is `0.0`, not `-0.0`.

## Constraints

- Do not change `p` or `q`.
- Use numpy only.

## Examples

```python
entropy(np.array([0.5, 0.5]))
entropy(np.array([1.0, 0.0]))
entropy(np.array([0.25, 0.25, 0.25, 0.25]))
entropy(np.array([0.5, 0.5]), base=np.e)
cross_entropy(np.array([0.5, 0.5]), np.array([0.9, 0.1]))
kl_divergence(np.array([0.5, 0.5]), np.array([0.9, 0.1]))
kl_divergence(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
```

```text
1.0
0.0
2.0
0.6931471805599453
1.736965594166206
0.7369655941662061
inf
```
