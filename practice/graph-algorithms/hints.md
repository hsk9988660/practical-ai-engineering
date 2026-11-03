## Hint 1

A breadth-first search takes nodes from the front of a queue. Start with the source in the queue and distance 0, and mark every other node as not visited with -1.

## Hint 2

When you take a node out of the queue, look at its neighbors. A neighbor with distance -1 is new: give it the distance of the current node plus 1 and add it to the end of the queue.

## Hint 3

A stack in place of a queue gives a depth-first order. It still visits every reachable node, but the distances it assigns can be too long.

## Hint 4

For the components, go through the nodes in order. When you find a node without a label, run a BFS from it and give every reached node the next label.

## Hint 5

The Laplacian needs the row sums of the adjacency matrix on the diagonal, and then the matrix is subtracted. `np.diag` builds a diagonal matrix from a vector.
