A neural network is a long chain of simple operations. Backpropagation applies the chain rule through that chain to get the gradient of the loss for every parameter. You build the smallest version of this idea: a class for one number that remembers how it was made and can send gradients back to its inputs.

## Task

Write the class `Value`. A `Value` holds a float `data` and a float `grad`, which starts at `0.0`.

Operations that make a new `Value` from other values:

| Expression | Result | Local derivative |
|---|---|---|
| `a + b` | $a + b$ | 1 for each operand |
| `a * b` | $a b$ | $b$ for `a`, and $a$ for `b` |
| `a ** n` | $a^n$, with `n` an int or float | $n a^{n-1}$ |
| `a.tanh()` | $\tanh a$ | $1 - \tanh^2 a$ |
| `a.relu()` | $\max(0, a)$ | 1 if $a > 0$, else 0 |
| `a.exp()` | $e^a$ | $e^a$ |
| `-a`, `a - b`, `a / b` | built from the operations above | |

Also support plain numbers on either side, for example `2 * a`, `a + 1`, `3 - a`.

The method `backward()` on the output value computes `grad` for every value that the output depends on:

1. Set the `grad` of the output to 1.
2. Visit all values that lead to the output in reverse order of creation (reverse topological order). For each one, add the local derivative times its own `grad` to the `grad` of each operand.

A value that is used in several places must **add up** the gradients that come from each place.

## Input and output

| Name | Type | Meaning |
|---|---|---|
| `Value(data)` | float or int | a leaf value. `data` becomes a float |
| `.data` | float | the number |
| `.grad` | float | the derivative of the output of the last `backward()` call with respect to this value |
| `.backward()` | method | fills in `grad` for the whole graph. It returns nothing |

## Constraints

- Use the standard library only. `math` is allowed.
- Do not call `backward()` twice on the same graph without making the values again. The tests do not do this.
- The graphs in the tests are small. A recursive visit of the graph is fine.

## Examples

```python
x = Value(2.0)
y = x * x + x * 3
y.backward()
y.data, x.grad
```

```text
(10.0, 7.0)
```

```python
p = Value(0.5)
q = (p.tanh() * 2 - 1) ** 2
q.backward()
q.data, p.grad
```

```text
(0.005740439096251332, -0.23834300632936067)
```
