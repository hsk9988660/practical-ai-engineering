import numpy as np

from practice_check import check, close, fixture, fixtures, shape


@check("examples", "two vectors in the plane", "The first output is the first input divided by its length. The second is the part of the second input that is at a right angle to it, also with length 1.")
def example_plane(sub):
    out = sub.gram_schmidt(np.array([[3.0, 1.0], [2.0, 2.0]]))
    close(out, [[0.9486833, 0.31622777], [-0.31622777, 0.9486833]], rtol=1e-6, what="the basis")


@check("examples", "a dependent vector is skipped", "The second row is 2 times the first, so nothing is left after the projection. Only 2 basis vectors remain.")
def example_dependent(sub):
    out = sub.gram_schmidt(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))
    shape(out, (2, 3), what="the basis")
    close(out, [[1.0, 0.0, 0.0], [0.0, 0.70710678, 0.70710678]], rtol=1e-6, atol=1e-12, what="the basis")


@check("properties", "the rows are orthonormal", "Q times Q transposed must be the identity: unit length on the diagonal and zero dot products elsewhere.")
def orthonormal(sub):
    rng = np.random.default_rng(0)
    out = sub.gram_schmidt(rng.normal(size=(4, 6)))
    close(out @ out.T, np.eye(4), atol=1e-9, what="Q Q^T")


@check("properties", "the basis spans every input vector", "Each input vector must equal the sum of its projections on the basis vectors.")
def spans_inputs(sub):
    rng = np.random.default_rng(1)
    v = rng.normal(size=(5, 5))
    out = sub.gram_schmidt(v)
    close((v @ out.T) @ out, v, atol=1e-9, what="the input rebuilt from the basis")


@check("properties", "the first row is the normalized first vector", "The first vector has nothing to subtract, so it is only divided by its length.")
def first_row(sub):
    v = np.array([[0.0, 3.0, 4.0], [1.0, 1.0, 1.0]])
    close(sub.gram_schmidt(v)[0], [0.0, 0.6, 0.8], what="the first row")


@check("properties", "the number of rows is the rank", "A set of 3 vectors in which the third is the sum of the first two has rank 2.")
def rank_count(sub):
    a = np.array([1.0, 2.0, 0.0, 1.0])
    b = np.array([0.0, 1.0, 3.0, -1.0])
    out = sub.gram_schmidt(np.array([a, b, a + b]))
    shape(out, (2, 4), what="the basis")


@check("properties", "the input is not changed", "The function must work on a copy. Writing into the input breaks the code that calls it.")
def input_unchanged(sub):
    v = np.array([[3.0, 1.0, 0.0], [2.0, 2.0, 1.0], [0.0, 1.0, 5.0]])
    saved = v.copy()
    sub.gram_schmidt(v)
    close(v, saved, atol=0, rtol=0, what="the input")


@check("edge", "a zero vector is skipped", "A zero vector has no direction. It must not become a row, and the code must not divide by zero.")
def zero_vector(sub):
    out = sub.gram_schmidt(np.array([[0.0, 0.0], [0.0, 2.0]]))
    close(out, [[0.0, 1.0]], what="the basis")


@check("edge", "all vectors are zero", "No vector has any length, so the basis is empty. The shape must be (0, d).")
def all_zero(sub):
    shape(sub.gram_schmidt(np.zeros((3, 4))), (0, 4), what="the basis")


@check("edge", "a vector that is almost dependent is skipped", "A residual of 1e-13 is below the tolerance 1e-10. It is rounding noise and not a new direction.")
def almost_dependent(sub):
    v = np.array([[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-13, 0.0]])
    shape(sub.gram_schmidt(v), (1, 3), what="the basis")


@check("edge", "more vectors than dimensions", "In 2 dimensions at most 2 vectors can be independent.")
def too_many(sub):
    v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -3.0]])
    out = sub.gram_schmidt(v)
    shape(out, (2, 2), what="the basis")
    close(out @ out.T, np.eye(2), atol=1e-9, what="Q Q^T")


@fixture("sets")
def sets(ref):
    rng = np.random.default_rng(42)
    for n, d, dependent in [(1, 3, False), (3, 3, False), (4, 7, False), (5, 4, True), (6, 6, True)]:
        v = rng.normal(size=(n, d))
        if dependent and n >= 3:
            v[2] = 2.0 * v[0] - v[1]
        yield {"v": v}, ref.gram_schmidt(v)


@check("reference", "the basis matches the reference", "Your rows are the same as the reference rows. The signs and the order are fixed by the method.")
def reference_basis(sub):
    for case, expected in fixtures("sets"):
        close(sub.gram_schmidt(case["v"]), expected, rtol=1e-7, atol=1e-9, what=f"the basis for an input of shape {case['v'].shape}")
