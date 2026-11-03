## Hint 1

Keep a Python list called `basis`. Each new vector is compared with the unit vectors in this list.

## Hint 2

The projection of $w$ on a unit vector $q$ is `np.dot(q, w) * q`. Because $q$ has length 1, you do not divide by $q \cdot q$.

## Hint 3

Take a copy of the row before you subtract. Update `w` after each projection, so that the next projection uses the new `w`.

## Hint 4

After the loop over the basis, check `np.linalg.norm(w) > tol`. Only then append `w / norm`. At the end, reshape the list to `(len(basis), d)` so that an empty basis has the shape `(0, d)`.
