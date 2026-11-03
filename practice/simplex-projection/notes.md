## Why a shift and a clip

The nearest point on the simplex solves a small optimization problem with constraints. Its optimality conditions say that all positive coordinates are shifted by the same amount, $\theta$, and that coordinates that would become negative are set to 0. The sort finds which coordinates stay positive.

## Cost

The sort makes the projection cost $O(n \log n)$. There is also a linear-time method, but the sort-based one is simple and fast enough for most uses.

## Projected gradient descent

For a convex function and a convex constraint set, projected gradient descent converges with a suitable learning rate. The projection is the only change from plain gradient descent. If the projection is cheap, as here, constrained problems are almost as easy as unconstrained ones.

## Where it appears

Mixture weights, attention-like weights that must sum to 1, optimal transport, portfolio optimization and the dual of many problems all live on a simplex. Some methods use the sparsemax function, which is exactly this projection, in place of the softmax: it can give exact zeros.

## Where this goes next

The projection onto a box is a clip, and the projection onto a ball is a rescaling. Gradient clipping and weight constraints in deep learning are projections of this kind.
