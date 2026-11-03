## Why this function is hard

The minimum is at (1, 1) and the loss there is 0. The valley along the curve $y = x^2$ is almost flat, but its walls are very steep. A learning rate that is small enough to be stable across the valley makes progress along the valley very slow. A larger rate jumps from wall to wall and diverges. Plain gradient descent shows both problems after a few steps.

## Always check a gradient

A wrong gradient does not crash. The loss just goes down more slowly, or not at all. A finite-difference check of every new gradient takes a few lines and finds sign errors and missing terms at once. The tests use it for this exercise.

## The update

The step is $p \leftarrow p - \eta \nabla f(p)$. The minus sign moves against the gradient, which is the direction of steepest descent. With a plus sign you get gradient ascent, and the loss grows.

## History

Recording the loss at every step is the simplest way to see how training behaves. A plot of the history shows slow progress, oscillation or divergence. The history has `steps + 1` entries because it includes the start point.

## Where this goes next

Momentum and Adam, in the next exercises of this phase, change the update rule so that the valley is crossed faster. The learning rate lesson in the deep learning core phase adds schedules for $\eta$.
