## The idea

Every operation knows its own local derivative. The chain rule says that the derivative of the output with respect to an early value is the product of the local derivatives along each path, summed over all paths. The backward pass computes this for all values at once. It visits each value one time, so it costs about as much as the forward pass.

## Why a sum

If a value feeds two later operations, a small change in it changes the output along two paths. The two effects add. This is why every update is `+=`. In the example `x * x`, the multiplication sends `x.data * out.grad` to `self` and the same to `other`, which is the same object. The two parts add up to $2x$.

## Why reverse topological order

A value must have the full gradient from all of its users before it sends gradients to its own inputs. In the reverse of a topological order, every user of a value is visited before the value itself. This is the only requirement. A graph with a diamond shape, or a long chain, goes wrong if the order is wrong.

## Forward mode and reverse mode

Reverse mode gets the gradient of one output with respect to all inputs in one backward pass. This is the case in machine learning: one loss, millions of parameters. Forward mode gets the derivatives of all outputs with respect to one input per pass, which is cheaper when there are few inputs and many outputs.

## What real libraries add

PyTorch and JAX do the same for tensors and not for single numbers. Each operation has a backward rule for whole arrays. They also free the graph after the backward pass, and they let you ask for the gradient of a gradient.

## Where this goes next

The next lessons use this idea to train networks: the backpropagation lesson in the deep learning core phase builds the same graph with matrices, and the mini framework lesson wraps it in layers and optimizers.
