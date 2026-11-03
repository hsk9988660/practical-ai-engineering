## Hint 1

Each `Value` stores its `data`, its `grad`, the set of values that it was made from, and a function `_backward` that sends the gradient of this value to those values. A leaf has a `_backward` that does nothing.

## Hint 2

In `__add__` and `__mul__`, first turn a plain number into a `Value`. Make the result with the operands as its children. Define `_backward` inside the method, so that it can see `self`, `other` and `out`, and set `out._backward` to it.

## Hint 3

Inside `_backward`, use `+=` for the gradient of every operand. A value that appears twice, as in `x * x`, receives two contributions, and `=` would keep only one of them. Multiply every local derivative by `out.grad`: this is the chain rule.

## Hint 4

For `backward()`, build a list of all values in topological order: a recursive function that visits the children first and then appends the value. Set `self.grad = 1.0`. Then call `_backward()` for the values in reversed order. Reverse order guarantees that the `grad` of a value is complete before it passes the gradient on.

## Hint 5

Build `-a`, `a - b` and `a / b` from the basic operations: `a * -1`, `a + (-b)` and `a * b ** -1`. Then they need no backward code of their own.
