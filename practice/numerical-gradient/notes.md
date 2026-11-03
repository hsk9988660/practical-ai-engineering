## Why central differences

A forward difference has an error that grows like $h$. A central difference has an error that grows like $h^2$, because the first error terms of the Taylor series cancel. For the same step it is much more exact, and it costs two function calls for each entry instead of one.

## Choosing h

If $h$ is too large, the slope is a bad estimate of the derivative. If $h$ is too small, $f(x+h) - f(x-h)$ is a difference of two almost equal numbers, and rounding error dominates. For float64, a value near $10^{-6}$ is a good start. Float32 needs a larger step, near $10^{-3}$.

## Reading the relative error

As a rough rule for float64 and smooth functions: below $10^{-7}$ the gradient is right, between $10^{-7}$ and $10^{-4}$ look closer, and above $10^{-3}$ there is probably a bug. The relative error does not depend on the size of the gradient, which is why it is better than the absolute difference. If the gradients are tiny, the absolute difference is always small and tells you nothing.

## The cost

You call `f` twice for every parameter. For a model with millions of parameters this is far too slow for training, so use the gradient check on a tiny model and a few inputs, and turn it off after that.

## Where this goes next

The autodiff lesson builds a backward pass automatically. The tests for it, and for the layers in the deep learning core phase, use this check. The practice harness has the same helper as `numeric_grad`.
