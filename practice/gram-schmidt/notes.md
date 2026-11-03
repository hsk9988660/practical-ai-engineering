## What the method does

Each new vector loses the parts that point along the earlier basis vectors. What is left is at a right angle to all of them. After you divide by the length, you have a new unit vector. The first $k$ output vectors span the same space as the first $k$ independent input vectors.

## Why this is QR

If you stack the input vectors as columns, the output vectors are the columns of $Q$ and the dot products that you computed are the entries of $R$ in the factorization $A = QR$. Libraries compute QR with Householder reflections, which are more stable, but the idea is the same.

## Modified and classical versions

The code subtracts the projection on the updated $w$ at each step. This is modified Gram-Schmidt. The classical version projects the original $v$ on every basis vector. They agree in exact arithmetic. With vectors that are almost dependent, the classical version loses orthogonality much faster in floating point.

## Rank detection

A vector that depends on the earlier ones leaves a residual of zero, up to rounding. The tolerance decides what counts as zero. The number of rows that you keep is the numerical rank of the input.

## Where this goes next

Orthogonal bases appear in PCA, in SVD, in the QR algorithm for eigenvalues, and in orthogonal weight initialization for neural networks.
