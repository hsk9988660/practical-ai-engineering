## The inverse CDF method

If $U$ is uniform on $[0, 1]$ and $F$ is a CDF, then $F^{-1}(U)$ has the distribution $F$. For the exponential distribution, the inverse is a simple formula. For the normal distribution there is no simple inverse, which is why Box-Muller exists.

## Box-Muller

Two independent normal numbers form a point in the plane. Its direction is uniform and its squared radius is exponential with rate 1/2. Box-Muller makes a point from a random radius and a random angle, and so turns two uniform numbers into two normal numbers.

## Categorical sampling

The cumulative sum cuts the interval $[0, 1)$ into pieces. Each piece has the length of one probability. The piece that contains $u$ gives the category. A category with probability 0 has a piece of length 0, so it is never chosen. The sampling step of a language model does the same on the probabilities of the next token.

## Why take the uniforms as input

When the uniform numbers are an input, the function is deterministic. This makes tests exact, makes experiments repeatable, and lets you use common random numbers to compare two methods fairly.

## Where this goes next

Top-k and top-p sampling, in the LLM inference phase, use a categorical sample after they filter the probabilities. Diffusion models use normal samples to add noise.
