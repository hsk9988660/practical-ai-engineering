## Why a queue

A queue serves the nodes in the order in which they were found. All nodes at distance 1 come out before any node at distance 2. So the first time that a node is reached, it is reached by a shortest path. With a stack, a node can be reached first by a long path and is never updated again, because it is marked as visited.

## Cost

Each node enters the queue once and each edge is looked at twice, so the search costs $O(n + m)$. The component labeling runs one search for each component, so it also costs $O(n + m)$ in total.

## The Laplacian

$L = D - A$ is symmetric and positive semi-definite. For a vector $x$, $x^\top L x$ is the sum of $(x_u - x_v)^2$ over all edges, so it measures how much $x$ changes along edges. The vector of ones always has eigenvalue 0. The number of zero eigenvalues is the number of connected components, and the second smallest eigenvalue measures how well the graph is connected.

## Where this goes next

Spectral clustering uses the eigenvectors of the Laplacian to split a graph into groups. Graph neural networks multiply node features by a normalized adjacency matrix, which is closely related to $L$. Search engines and recommenders use graph distances and walks.
