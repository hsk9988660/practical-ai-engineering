## What the numbers mean

A fair coin has an entropy of 1 bit: you need one yes/no question on average to learn the outcome. A distribution that is sure of the outcome has 0 bits. The uniform distribution over $n$ outcomes has the largest entropy, $\log_2 n$.

## Cross-entropy as a loss

When $p$ is a one-hot label, the cross-entropy is $-\log q_{\text{correct}}$. This is the classification loss in the deep learning phases. Minimizing it over $q$ is the same as minimizing the KL divergence, because $H(p)$ does not depend on $q$.

## KL divergence is not a distance

It is never negative, and it is 0 only when $p = q$. But it is not symmetric: $D_{KL}(p \| q) \ne D_{KL}(q \| p)$ in general. Its value is infinite when $q$ gives probability 0 to an outcome that $p$ can produce. This is a real effect: a model that is sure that something is impossible pays an infinite price when it happens.

## Zero probabilities

The convention $0 \log 0 = 0$ comes from the limit of $p \log p$ as $p$ goes to 0. In code you cannot let numpy evaluate the product, because it gives NaN. A mask is the clean solution. Do not add a small constant to hide zeros when you want an exact answer.

## Base

Base 2 gives bits and base $e$ gives nats. Deep learning libraries use nats. To convert, multiply nats by $1 / \ln 2$ to get bits.

## Where this goes next

Perplexity is $e^{H}$ of a language model's cross-entropy. The KL divergence appears in variational autoencoders, in RLHF as a penalty that keeps a model near a reference, and in knowledge distillation.
