## The shift trick

Subtracting the maximum does not change the result of the log-sum-exp, because $\log \sum e^{x_i} = m + \log \sum e^{x_i - m}$. It moves the largest exponent to 0, so the largest term is exactly 1. The sum is then between 1 and $n$, which is safe.

## log1p

For a tiny $y$, `1 + y` rounds to 1 and the information in $y$ is lost. `np.log1p(y)` computes $\log(1 + y)$ without forming `1 + y`. This is why `softplus(-40)` keeps its value of $4 \cdot 10^{-18}$.

## Never take the log of a softmax

A softmax entry can underflow to 0, and the log of 0 is minus infinity. `log_softmax` computes the log-probabilities directly. This is also why frameworks have a fused cross-entropy that takes logits and not probabilities.

## Minus infinity

Masks in attention use $-\infty$ for the positions that must not be seen. A row of only $-\infty$ is an edge case that the code must handle on purpose. Here the answer is $-\infty$, which says that the total probability mass is zero.

## Where this goes next

The softmax and cross-entropy exercise, the attention exercises and the loss functions in the later phases use these forms. Mixed precision training makes them more important, because float16 overflows at about 65000.
