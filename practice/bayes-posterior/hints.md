## Hint 1

The numerator is an element-wise product of the prior and the likelihood. The denominator is the sum of that product.

## Hint 2

Check the sum before you divide. A sum of 0 means that no hypothesis can explain the data.

## Hint 3

`np.asarray(prior, dtype=float) * ...` makes a new array. Do not use `*=` on the array that the caller gave you.

## Hint 4

For many observations, loop over the rows of `likelihoods` and call `posterior` again with the last result as the new prior. Each call rescales the numbers to add up to 1, so they never become too small.
