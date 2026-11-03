Graphs describe networks: friends, links between pages, atoms in a molecule, the layers of a model. Two basic questions are how far one node is from another and which nodes belong together. The graph Laplacian is a matrix that links these questions to linear algebra, and it is the starting point of spectral clustering and of graph neural networks.

## Task

A graph is undirected and unweighted. It has nodes `0` to `n - 1`. It is given as an adjacency list `adj`, a list where `adj[u]` is the list of the neighbors of `u`. If `v` is in `adj[u]`, then `u` is in `adj[v]`.

Write three functions.

1. `bfs_distances(adj, source)` returns a list with the number of edges on a shortest path from `source` to every node. Use a breadth-first search with a queue. A node that cannot be reached gets `-1`.
2. `connected_components(adj)` returns a list `labels` with one label for each node. Nodes in the same component have the same label. The labels are `0, 1, 2, ...`, assigned in the order of the smallest node of each component.
3. `laplacian(a)` takes an adjacency **matrix** `a` (a 0/1 array of shape `(n, n)`) and returns $L = D - A$, where $D$ is the diagonal matrix of the row sums of `a`.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `adj` | list of lists of int | `n` lists | the neighbors of each node |
| `source` | int | scalar | the start node |
| `a` | float or int array | `(n, n)` | the adjacency matrix of an undirected graph |

- `bfs_distances` returns a list of `n` ints. `connected_components` returns a list of `n` ints.
- `laplacian` returns a float array with the shape `(n, n)`.

## Constraints

- Do not change `adj` or `a`.
- Use numpy and the standard library only (`collections.deque` is a good queue).

## Examples

```python
adj = [[1], [0, 2], [1], []]
bfs_distances(adj, 0)
connected_components(adj)
laplacian(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
```

```text
[0, 1, 2, -1]
[0, 0, 0, 1]
array([[ 1., -1.,  0.],
       [-1.,  2., -1.],
       [ 0., -1.,  1.]])
```
