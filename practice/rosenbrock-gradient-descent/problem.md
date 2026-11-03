The Rosenbrock function has a long, narrow, curved valley. Its minimum is easy to see and hard to reach, so it is a standard test for optimizers. You write the function, its gradient and the simplest optimizer, gradient descent.

## Task

Write three functions for the point $p = (x, y)$.

1. `rosenbrock(p)` returns the loss

$$f(x, y) = (1 - x)^2 + 100\,(y - x^2)^2.$$

2. `rosenbrock_grad(p)` returns the gradient

$$\frac{\partial f}{\partial x} = -2(1 - x) - 400\,x\,(y - x^2), \qquad \frac{\partial f}{\partial y} = 200\,(y - x^2)$$

as an array `[df/dx, df/dy]`.

3. `gradient_descent(start, lr, steps)` repeats `steps` times: $p \leftarrow p - \text{lr} \cdot \nabla f(p)$. It returns `(p, history)`. `p` is the final point. `history` is a list of the loss at the start point and after every step, so it has `steps + 1` numbers.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `p` | float array | `(2,)` | the point $(x, y)$ |
| `start` | float array | `(2,)` | the first point |
| `lr` | float | scalar | the learning rate |
| `steps` | int | scalar | the number of updates, at least 0 |

- `rosenbrock` returns a float.
- `rosenbrock_grad` returns a float array with the shape `(2,)`.
- `gradient_descent` returns a float array with the shape `(2,)` and a list of `steps + 1` floats.

## Constraints

- Do not change `start`.
- Use numpy only.

## Examples

```python
rosenbrock(np.array([1.0, 1.0]))
rosenbrock_grad(np.array([-1.0, 1.0]))
point, history = gradient_descent(np.array([-1.0, 1.0]), lr=0.001, steps=100)
point
history[0], round(history[-1], 6), len(history)
```

```text
0.0
array([-4.,  0.])
array([-0.91265103,  0.84098998])
(4.0, 3.664727, 101)
```
