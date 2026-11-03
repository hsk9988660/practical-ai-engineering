A formula that is correct on paper can fail in floating point. `exp(1000)` overflows to infinity, and `log(1 + exp(-40))` rounds to 0, although the true value is $4 \cdot 10^{-18}$. The functions below appear in almost every model, and each has a safe form. A model that uses the unsafe form works on small test inputs and then produces NaN in training.

## Task

Write four functions. Each one works on arrays of any shape and gives the same result as the textbook formula wherever that formula works.

1. `sigmoid(x)` returns $1 / (1 + e^{-x})$.
2. `softplus(x)` returns $\log(1 + e^{x})$.
3. `logsumexp(x, axis=-1)` returns $\log \sum_i e^{x_i}$ along `axis`. The axis is removed from the shape. If all values on the axis are $-\infty$, the result is $-\infty$.
4. `log_softmax(x, axis=-1)` returns $x_i - \operatorname{logsumexp}(x)$ along `axis`, with the shape of `x`.

None of the functions may raise a warning or return NaN for finite inputs of any size, for example 1000 or -1000.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `x` | float array | any shape | the input values |
| `axis` | int | scalar | the axis of the reduction, default the last axis |

- `sigmoid` and `softplus` return an array with the shape of `x`.
- `logsumexp` returns an array without the reduced axis. `log_softmax` returns the shape of `x`.

## Constraints

- Do not change `x`.
- Do not use `scipy`. Use numpy only.

## Examples

```python
sigmoid(np.array([-1000.0, 0.0, 1000.0]))
softplus(np.array([-40.0, 0.0, 1000.0]))
float(logsumexp(np.array([1000.0, 1000.0])))
log_softmax(np.array([1000.0, 1001.0]))
```

```text
array([0. , 0.5, 1. ])
array([4.24835426e-18, 6.93147181e-01, 1.00000000e+03])
1000.6931471805599
array([-1.31326169, -0.31326169])
```
