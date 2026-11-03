Bayes' rule tells you how to change your belief about several hypotheses when you see data. You start with a prior for each hypothesis, you multiply it by how well the hypothesis explains the data, and you rescale so that the beliefs add up to 1.

## Task

Write two functions.

1. `posterior(prior, likelihood)` returns the posterior over the hypotheses:

$$P(h \mid d) = \frac{P(d \mid h)\,P(h)}{\sum_{h'} P(d \mid h')\,P(h')}$$

Here `prior[h]` is $P(h)$ and `likelihood[h]` is $P(d \mid h)$ for one observation $d$. If the sum in the denominator (the evidence) is 0, raise `ValueError`.

2. `sequential_posterior(prior, likelihoods)` handles many independent observations. `likelihoods[t, h]` is the likelihood of observation $t$ under hypothesis $h$. The posterior after observation $t$ is the prior for observation $t + 1$. Return the posterior after the last observation. With no observations, return the prior as a float array.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `prior` | float array | `(H,)` | the belief in each hypothesis, adding up to 1 |
| `likelihood` | float array | `(H,)` | $P(d \mid h)$ for one observation |
| `likelihoods` | float array | `(T, H)` | one row of likelihoods for each observation |

- Both functions return a float array with the shape `(H,)` that adds up to 1.

## Constraints

- Do not change `prior` or `likelihoods`.
- Normalize after every observation in `sequential_posterior`. A product of 1000 small likelihoods is too small for a float.
- Use numpy only.

## Examples

```python
posterior(np.array([0.01, 0.99]), np.array([0.95, 0.05]))
sequential_posterior(np.array([0.5, 0.5]), np.array([[0.9, 0.2]] * 3))
```

```text
array([0.16101695, 0.83898305])
array([0.98914518, 0.01085482])
```

The first example is a medical test. A disease has a prior of 1 percent. The test is positive for 95 percent of sick people and for 5 percent of healthy people. After a positive test, the chance of the disease is only about 16 percent.
