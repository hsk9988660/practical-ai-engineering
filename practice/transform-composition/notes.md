## Order matters

Matrix multiplication is not commutative. Scaling the x axis by 2 and then shearing gives another result than shearing and then scaling, as the two examples show. When you read a product $C B A$ from the right, it says: apply $A$, then $B$, then $C$. A graphics or robotics pipeline that gets this order wrong draws the right shapes at the wrong places.

## One matrix for a whole pipeline

You multiply the matrices once and then apply the result to a million points. This costs one matrix product and not one product for each step and each point.

## Row points and column points

Many libraries store a point as a row. Then the formula flips: $P' = P M^\top$. If your data has the points in rows, the transpose is needed. Without it, rotation turns the points the wrong way and other matrices give wrong results.

## Useful facts that the tests use

- A rotation matrix is orthogonal: its transpose is its inverse. Its determinant is 1, so it keeps areas.
- The determinant of a product is the product of the determinants. Scaling by $s_x$ and $s_y$ changes areas by $s_x s_y$.
- Two rotations add their angles: $R(a) R(b) = R(a + b)$.

## Where this goes next

The same idea appears in the layers of a neural network, which are matrices that transform vectors one after the other. Computer vision uses it for image rotation, scaling and augmentation, and in homogeneous coordinates for translation and camera models.
