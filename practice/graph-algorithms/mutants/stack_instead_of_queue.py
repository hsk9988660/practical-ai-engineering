from collections import deque

import numpy as np


def bfs_distances(adj, source):
    dist = [-1] * len(adj)
    dist[source] = 0
    queue = deque([source])
    while queue:
        node = queue.pop()
        for neighbor in adj[node]:
            if dist[neighbor] < 0:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    return dist


def connected_components(adj):
    labels = [-1] * len(adj)
    count = 0
    for start in range(len(adj)):
        if labels[start] < 0:
            for node, d in enumerate(bfs_distances(adj, start)):
                if d >= 0:
                    labels[node] = count
            count += 1
    return labels


def laplacian(a):
    a = np.asarray(a, dtype=float)
    return np.diag(a.sum(axis=1)) - a
