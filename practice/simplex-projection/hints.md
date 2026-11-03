## Hint 1

Check the answer on small vectors by hand. For `[2, 0]` the nearest point of the simplex is the corner `[1, 0]`.

## Hint 2

The cumulative sum `css` has `z` subtracted. The condition `u - css / index > 0` is true for the first `rho` positions and false after that, so the last true index is `rho`.

## Hint 3

`index[condition][-1]` gives `rho`. Because `rho` counts from 1, the matching entry of `css` is `css[rho - 1]`.

## Hint 4

Once you have `theta`, the projection is `np.maximum(v - theta, 0.0)`.

## Hint 5

`projected_gradient` is a short loop. Project the start point too, so that a run with 0 steps still returns a feasible point.
