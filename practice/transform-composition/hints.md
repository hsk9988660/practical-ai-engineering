## Hint 1

The first three functions only build a small array. Use the formulas in the task.

## Hint 2

If you apply `A` first and `B` second, a point $p$ becomes $B(Ap) = (BA)p$. The matrix of the first transformation is on the right of the product.

## Hint 3

For `compose`, start with the identity matrix `np.eye(2)`. For each matrix in the order of the arguments, update `result = m @ result`.

## Hint 4

`points` has one point in each row, but the matrix acts on columns. For all points at once, compute `points @ matrix.T`.
