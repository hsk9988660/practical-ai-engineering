import numpy as np

from practice_check import check, close, fixture, fixtures, shape, true

PATH_PLUS_ISOLATED = [[1], [0, 2], [1], []]


def to_matrix(adj):
    m = np.zeros((len(adj), len(adj)))
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            m[u, v] = 1.0
    return m


def random_graph(rng, n, p):
    upper = np.triu(rng.random((n, n)) < p, 1)
    sym = upper | upper.T
    return [list(np.nonzero(sym[i])[0]) for i in range(n)]


@check("examples", "distances on a path with an isolated node", "Node 3 has no edge, so it cannot be reached and its distance is -1.")
def example_bfs(sub):
    true(sub.bfs_distances(PATH_PLUS_ISOLATED, 0) == [0, 1, 2, -1], "the distances are wrong", [0, 1, 2, -1], sub.bfs_distances(PATH_PLUS_ISOLATED, 0))


@check("examples", "components of the same graph", "Nodes 0, 1 and 2 are connected and get label 0. The isolated node 3 gets label 1.")
def example_components(sub):
    true(sub.connected_components(PATH_PLUS_ISOLATED) == [0, 0, 0, 1], "the labels are wrong", [0, 0, 0, 1], sub.connected_components(PATH_PLUS_ISOLATED))


@check("examples", "Laplacian of a path with 3 nodes", "The Laplacian is the degree matrix minus the adjacency matrix. The degrees are 1, 2 and 1.")
def example_laplacian(sub):
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    close(sub.laplacian(a), [[1, -1, 0], [-1, 2, -1], [0, -1, 1]], atol=0, what="the Laplacian")


@check("properties", "BFS finds the shortest path, also when there is a shortcut", "In a cycle 0-1-2-3-0, node 3 is one step from node 0. A search that goes deep first finds the long way around.")
def shortcut(sub):
    cycle = [[1, 3], [0, 2], [1, 3], [2, 0]]
    true(sub.bfs_distances(cycle, 0) == [0, 1, 2, 1], "the distances are wrong", [0, 1, 2, 1], sub.bfs_distances(cycle, 0))


@check("properties", "the source has distance 0 and neighbors differ by at most 1", "In an unweighted graph, two neighbors cannot be more than one step apart from the source.")
def distance_properties(sub):
    rng = np.random.default_rng(0)
    adj = random_graph(rng, 12, 0.25)
    dist = sub.bfs_distances(adj, 3)
    true(dist[3] == 0, "the source distance is not 0", 0, dist[3])
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            if dist[u] >= 0:
                true(dist[v] >= 0 and abs(dist[u] - dist[v]) <= 1, "neighbors differ by more than 1", "at most 1", (u, v, dist[u], dist[v]))


@check("properties", "BFS distances match a matrix power search", "The distance to a node is the smallest k for which the k-th power of the adjacency matrix has a non-zero entry.")
def matches_matrix_power(sub):
    rng = np.random.default_rng(1)
    adj = random_graph(rng, 9, 0.3)
    a = to_matrix(adj)
    expected = [-1] * 9
    expected[0] = 0
    power = np.eye(9)
    for k in range(1, 9):
        power = power @ a
        for v in range(9):
            if expected[v] < 0 and power[0, v] > 0:
                expected[v] = k
    true(sub.bfs_distances(adj, 0) == expected, "the distances differ from the matrix-power search", expected, sub.bfs_distances(adj, 0))


@check("properties", "component labels are 0, 1, 2, ... in order of the first node", "Each component gets the next unused label, in the order of the smallest node that it contains.")
def label_order(sub):
    adj = [[2], [3], [0], [1], []]
    true(sub.connected_components(adj) == [0, 1, 0, 1, 2], "the labels are wrong", [0, 1, 0, 1, 2], sub.connected_components(adj))


@check("properties", "the number of zero eigenvalues of the Laplacian is the number of components", "This is a basic fact of spectral graph theory.")
def laplacian_components(sub):
    rng = np.random.default_rng(2)
    for _ in range(6):
        adj = random_graph(rng, 10, 0.15)
        labels = sub.connected_components(adj)
        eigenvalues = np.linalg.eigvalsh(sub.laplacian(to_matrix(adj)))
        true(int(np.sum(eigenvalues < 1e-9)) == len(set(labels)), "the zero eigenvalues do not match the components", len(set(labels)), int(np.sum(eigenvalues < 1e-9)))


@check("properties", "the Laplacian is symmetric, has rows that sum to 0, and is positive semi-definite", "These properties follow from L = D - A for an undirected graph.")
def laplacian_properties(sub):
    rng = np.random.default_rng(3)
    a = to_matrix(random_graph(rng, 8, 0.4))
    lap = sub.laplacian(a)
    shape(lap, (8, 8), what="the Laplacian")
    close(lap, lap.T, atol=0, what="the Laplacian against its transpose")
    close(lap.sum(axis=1), np.zeros(8), atol=1e-12, what="the row sums")
    true(bool(np.all(np.linalg.eigvalsh(lap) > -1e-9)), "an eigenvalue is negative", "all eigenvalues at least 0", np.linalg.eigvalsh(lap).min())
    close(np.diag(lap), a.sum(axis=1), atol=0, what="the diagonal (the degrees)")


@check("properties", "the input is not changed", "The functions must not write into the graph.")
def input_unchanged(sub):
    adj = [[1, 2], [0], [0]]
    a = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    saved = ([list(x) for x in adj], a.copy())
    sub.bfs_distances(adj, 0)
    sub.connected_components(adj)
    sub.laplacian(a)
    true(adj == saved[0], "the adjacency list changed", saved[0], adj)
    close(a, saved[1], atol=0, rtol=0, what="the matrix")


@check("edge", "a single node and an empty graph", "One node has distance 0 to itself and forms one component. An empty graph has no labels.")
def tiny(sub):
    true(sub.bfs_distances([[]], 0) == [0], "the distance list is wrong", [0], sub.bfs_distances([[]], 0))
    true(sub.connected_components([[]]) == [0], "the labels are wrong", [0], sub.connected_components([[]]))
    true(sub.connected_components([]) == [], "an empty graph must give an empty list", [], sub.connected_components([]))


@check("edge", "a graph with no edges", "Every node is its own component. The Laplacian is the zero matrix.")
def no_edges(sub):
    true(sub.connected_components([[], [], []]) == [0, 1, 2], "each node must be its own component", [0, 1, 2], sub.connected_components([[], [], []]))
    close(sub.laplacian(np.zeros((3, 3))), np.zeros((3, 3)), atol=0, what="the Laplacian")


@check("edge", "a complete graph", "Every node is one step from every other node, and all nodes form one component.")
def complete(sub):
    n = 6
    adj = [[j for j in range(n) if j != i] for i in range(n)]
    true(sub.bfs_distances(adj, 2) == [1, 1, 0, 1, 1, 1], "the distances are wrong", [1, 1, 0, 1, 1, 1], sub.bfs_distances(adj, 2))
    true(sub.connected_components(adj) == [0] * n, "the labels are wrong", [0] * n, sub.connected_components(adj))


@fixture("graphs")
def graphs(ref):
    rng = np.random.default_rng(42)
    for n, p in [(5, 0.3), (10, 0.15), (15, 0.1), (20, 0.2)]:
        adj = random_graph(rng, n, p)
        adj = [[int(v) for v in nbrs] for nbrs in adj]
        yield {"adj": adj}, {"dist": ref.bfs_distances(adj, 0), "labels": ref.connected_components(adj), "lap": ref.laplacian(to_matrix(adj))}


@check("reference", "distances, components and Laplacian match the reference", "Your three functions agree with the reference on random graphs.")
def reference_all(sub):
    for case, expected in fixtures("graphs"):
        adj = [list(x) for x in case["adj"]]
        true(sub.bfs_distances(adj, 0) == expected["dist"], "the distances differ from the reference", expected["dist"], sub.bfs_distances(adj, 0))
        true(sub.connected_components(adj) == expected["labels"], "the labels differ from the reference", expected["labels"], sub.connected_components(adj))
        close(sub.laplacian(to_matrix(adj)), expected["lap"], atol=0, what="the Laplacian")
