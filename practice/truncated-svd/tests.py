import numpy as np

from practice_check import check, close, fixture, fixtures, shape, true

A = np.array([[3.0, 0.0], [0.0, 1.0]])


@check("examples", "the best rank-1 approximation of diag(3, 1)", "Keep only the largest singular value 3 and drop the value 1.")
def example_low_rank(sub):
    close(sub.low_rank(A, 1), [[3.0, 0.0], [0.0, 0.0]], atol=1e-9, what="the approximation")


@check("examples", "the energy kept by one component", "The energy is the sum of squared singular values. Here it is 9 of 10.")
def example_energy(sub):
    close(sub.energy_ratio(A, 1), 0.9, rtol=1e-12, what="the energy ratio")


@check("examples", "the smallest rank for an energy target", "Rank 1 keeps 90 percent. A target of 85 percent needs rank 1, and a target of 95 percent needs rank 2.")
def example_rank(sub):
    true(sub.smallest_rank(A, 0.85) == 1, "the rank for 85 percent is wrong", 1, sub.smallest_rank(A, 0.85))
    true(sub.smallest_rank(A, 0.95) == 2, "the rank for 95 percent is wrong", 2, sub.smallest_rank(A, 0.95))


@check("properties", "the approximation has rank k and the shape of the input", "A rank k approximation has k non-zero singular values and keeps the shape (m, n).")
def rank_and_shape(sub):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(7, 5))
    for k in (1, 2, 4):
        out = sub.low_rank(a, k)
        shape(out, (7, 5), what="the approximation")
        true(np.linalg.matrix_rank(out, tol=1e-9) == k, "the rank of the approximation is not k", k, np.linalg.matrix_rank(out, tol=1e-9))


@check("properties", "the error is the dropped singular values (Eckart-Young)", "The Frobenius error of the best rank k approximation is the square root of the sum of the squared dropped singular values.")
def eckart_young(sub):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(8, 6))
    s = np.linalg.svd(a, compute_uv=False)
    for k in (1, 3, 5):
        close(np.linalg.norm(a - sub.low_rank(a, k)), np.sqrt(np.sum(s[k:] ** 2)), rtol=1e-9, what=f"the error for k={k}")


@check("properties", "no random rank-k matrix is closer", "The truncated SVD is the best rank k approximation, so a random rank k matrix cannot beat it.")
def is_best(sub):
    rng = np.random.default_rng(2)
    a = rng.normal(size=(6, 6))
    best = np.linalg.norm(a - sub.low_rank(a, 2))
    for _ in range(20):
        other = rng.normal(size=(6, 2)) @ rng.normal(size=(2, 6))
        true(np.linalg.norm(a - other) >= best, "a random matrix beat the approximation", "a larger error than the SVD", np.linalg.norm(a - other))


@check("properties", "all singular values give back the matrix", "With k = min(m, n) nothing is dropped.")
def full_rank(sub):
    rng = np.random.default_rng(3)
    for m, n in [(5, 3), (3, 5), (4, 4)]:
        a = rng.normal(size=(m, n))
        close(sub.low_rank(a, min(m, n)), a, atol=1e-9, what=f"the reconstruction of a {m}x{n} matrix")


@check("properties", "energy ratio grows with k and ends at 1", "Every singular value adds energy. Keeping all of them keeps all of the energy.")
def energy_monotone(sub):
    rng = np.random.default_rng(4)
    a = rng.normal(size=(6, 4))
    values = [sub.energy_ratio(a, k) for k in range(1, 5)]
    true(all(b >= a_ - 1e-12 for a_, b in zip(values, values[1:])), "the energy ratio went down", "a non-decreasing list", values)
    close(values[-1], 1.0, rtol=1e-12, what="the energy ratio for all components")


@check("properties", "the input is not changed", "A function that writes into its input arrays breaks the code that calls it.")
def input_unchanged(sub):
    a = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0]])
    saved = a.copy()
    sub.low_rank(a, 1)
    sub.energy_ratio(a, 1)
    sub.smallest_rank(a, 0.9)
    close(a, saved, atol=0, rtol=0, what="the input")


@check("edge", "k = 0 gives the zero matrix", "No components means no information. The shape stays the same.")
def k_zero(sub):
    out = sub.low_rank(np.ones((3, 2)), 0)
    shape(out, (3, 2), what="the approximation")
    close(out, np.zeros((3, 2)), atol=0, what="the approximation")


@check("edge", "a rank-1 matrix needs one component", "A matrix that is an outer product has one non-zero singular value.")
def rank_one(sub):
    a = np.outer([1.0, 2.0, 3.0], [4.0, 5.0])
    close(sub.energy_ratio(a, 1), 1.0, rtol=1e-12, what="the energy ratio")
    true(sub.smallest_rank(a, 1.0) == 1, "a rank-1 matrix needs rank 1", 1, sub.smallest_rank(a, 1.0))
    close(sub.low_rank(a, 1), a, atol=1e-9, what="the approximation")


@check("edge", "the zero matrix", "Nothing is lost when you drop components of a zero matrix, so the energy ratio is 1.0. The smallest rank is 0.")
def zero_matrix(sub):
    z = np.zeros((3, 3))
    close(sub.energy_ratio(z, 1), 1.0, what="the energy ratio")
    true(sub.smallest_rank(z, 0.9) == 0, "the zero matrix has rank 0", 0, sub.smallest_rank(z, 0.9))


@fixture("matrices")
def matrices(ref):
    rng = np.random.default_rng(42)
    for m, n, k in [(4, 4, 2), (6, 3, 2), (3, 6, 1), (8, 8, 5)]:
        a = rng.normal(size=(m, n))
        yield {"a": a, "k": k}, {"low": ref.low_rank(a, k), "energy": ref.energy_ratio(a, k), "rank": ref.smallest_rank(a, 0.9)}


@check("reference", "approximation, energy and rank match the reference", "Your three functions agree with the reference on random matrices of several shapes.")
def reference_all(sub):
    for case, expected in fixtures("matrices"):
        a = np.array(case["a"])
        close(sub.low_rank(a, case["k"]), expected["low"], rtol=1e-8, atol=1e-10, what=f"the approximation of a {a.shape} matrix")
        close(sub.energy_ratio(a, case["k"]), expected["energy"], rtol=1e-10, what="the energy ratio")
        true(sub.smallest_rank(a, 0.9) == expected["rank"], "the smallest rank is wrong", expected["rank"], sub.smallest_rank(a, 0.9))
