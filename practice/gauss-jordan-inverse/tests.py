import numpy as np

from practice_check import check, close, fixture, fixtures, raises, shape


@check("examples", "a 2x2 inverse", "For [[2, 1], [5, 3]] the determinant is 1, so the inverse is [[3, -1], [-5, 2]].")
def example_2x2(sub):
    close(sub.inverse(np.array([[2.0, 1.0], [5.0, 3.0]])), [[3.0, -1.0], [-5.0, 2.0]], atol=1e-12, what="the inverse")


@check("examples", "a zero on the diagonal needs a row swap", "The matrix that swaps two coordinates is its own inverse. The diagonal entries are 0, so the code must swap rows.")
def example_swap(sub):
    close(sub.inverse(np.array([[0.0, 1.0], [1.0, 0.0]])), [[0.0, 1.0], [1.0, 0.0]], atol=1e-12, what="the inverse")


@check("examples", "a singular matrix raises ValueError", "The second row is 2 times the first, so no inverse exists.")
def example_singular(sub):
    raises(ValueError, sub.inverse, np.array([[1.0, 2.0], [2.0, 4.0]]))


@check("properties", "A times its inverse is the identity", "By the definition of the inverse, A A^-1 = I and A^-1 A = I.")
def product_identity(sub):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 5)) + 5.0 * np.eye(5)
    inv = sub.inverse(a)
    shape(inv, (5, 5), what="the inverse")
    close(a @ inv, np.eye(5), atol=1e-9, what="A times the inverse")
    close(inv @ a, np.eye(5), atol=1e-9, what="the inverse times A")


@check("properties", "the inverse of the inverse is the matrix", "Inverting twice gives back the matrix.")
def double_inverse(sub):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 4)) + 4.0 * np.eye(4)
    close(sub.inverse(sub.inverse(a)), a, atol=1e-8, what="the double inverse")


@check("properties", "the inverse of a product is the reversed product of inverses", "(A B)^-1 = B^-1 A^-1. The order flips.")
def product_rule(sub):
    rng = np.random.default_rng(2)
    a = rng.normal(size=(3, 3)) + 3.0 * np.eye(3)
    b = rng.normal(size=(3, 3)) + 3.0 * np.eye(3)
    close(sub.inverse(a @ b), sub.inverse(b) @ sub.inverse(a), atol=1e-9, what="the inverse of the product")


@check("properties", "the inverse of a diagonal matrix has the reciprocals", "For a diagonal matrix the inverse is diagonal with 1 / d on the diagonal.")
def diagonal(sub):
    d = np.array([2.0, -4.0, 0.5])
    close(sub.inverse(np.diag(d)), np.diag(1.0 / d), atol=1e-12, what="the inverse")


@check("properties", "the input is not changed", "The function must work on a copy. Writing into the input breaks the code that calls it.")
def input_unchanged(sub):
    a = np.array([[4.0, 3.0, 2.0], [1.0, 3.0, 1.0], [2.0, 1.0, 5.0]])
    saved = a.copy()
    sub.inverse(a)
    close(a, saved, atol=0, rtol=0, what="the input")


@check("edge", "a 1x1 matrix", "The inverse of [[a]] is [[1 / a]].")
def one_by_one(sub):
    close(sub.inverse(np.array([[4.0]])), [[0.25]], what="the inverse")


@check("edge", "a permutation matrix", "A permutation matrix has zeros on the diagonal in several places. Its inverse is its transpose.")
def permutation(sub):
    p = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    close(sub.inverse(p), p.T, atol=1e-12, what="the inverse")


@check("edge", "a tiny pivot needs a swap for accuracy", "The entry 1e-9 is not zero, but dividing by it makes the rounding errors 1e9 times larger. Choose the largest entry in the column.")
def tiny_pivot(sub):
    a = np.array([[1e-9, 1.0], [1.0, 1.0]])
    expected = np.array([[1.0, -1.0], [-1.0, 1e-9]]) / (1e-9 - 1.0)
    close(sub.inverse(a), expected, rtol=1e-9, atol=1e-12, what="the inverse")


@check("edge", "singular matrices raise ValueError", "A matrix with a dependent row, or a matrix of zeros, has no inverse.")
def singular_cases(sub):
    raises(ValueError, sub.inverse, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]))
    raises(ValueError, sub.inverse, np.zeros((3, 3)))


@fixture("matrices")
def matrices(ref):
    rng = np.random.default_rng(42)
    for n in (1, 2, 3, 4, 6, 8):
        a = rng.normal(size=(n, n))
        yield {"a": a}, ref.inverse(a)


@check("reference", "the inverse matches the reference", "Your inverse equals the reference inverse for random matrices of sizes 1 to 8.")
def reference_inverse(sub):
    for case, expected in fixtures("matrices"):
        close(sub.inverse(case["a"]), expected, rtol=1e-7, atol=1e-8, what=f"the inverse of a {case['a'].shape[0]}x{case['a'].shape[0]} matrix")
