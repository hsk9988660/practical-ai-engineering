## Hint 1

Unpack the point with `x, y = p`. Both `rosenbrock` and `rosenbrock_grad` are short formulas in `x` and `y`.

## Hint 2

For the derivative with respect to $x$, use the chain rule on both terms. The second term gives $2 \cdot 100\,(y - x^2) \cdot (-2x) = -400\,x\,(y - x^2)$. If you are not sure, compare your gradient with a finite-difference estimate at a few points.

## Hint 3

Copy the start point with `np.array(start, dtype=float)`. Then build a new array in every step with `p = p - lr * grad`.

## Hint 4

Put the loss of the start point in `history` before the loop. Add the loss after every update. With `steps = 0` the history has one number.
