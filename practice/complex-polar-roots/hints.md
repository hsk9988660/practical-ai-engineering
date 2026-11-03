## Hint 1

Convert the input with `np.asarray(z, dtype=complex)`. Then `z.real` and `z.imag` are float arrays, also for real input.

## Hint 2

`np.arctan2(y, x)` gives the angle in the right quadrant for every sign of $x$ and $y$. `np.arctan(y / x)` divides by zero for $x = 0$ and confuses opposite quadrants.

## Hint 3

The two functions that build a number from a radius and an angle share one formula. Write `from_polar` once and use it in `complex_power`, `roots_of_unity` and `nth_roots`.

## Hint 4

For the roots of $z$, all roots have the same radius $r^{1/n}$. The angles differ by $2\pi/n$, because adding a full turn $2\pi k$ to $\theta$ does not change $z$, but it changes $\theta / n$.
