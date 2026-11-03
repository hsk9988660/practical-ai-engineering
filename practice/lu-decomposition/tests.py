import numpy as np

from practice_check import check, close, fixture, fixtures, raises, shape, true

A = np.array([[2.0, 1.0], [4.0, 3.0]])


@check("examples", "LU of a 2x2 matrix with a row swap", "The largest entry of the first column is 4, so rows 0 and 1 swap. Then L = [[1, 0], [0.5, 1]] and U = [[4, 3], [0, -0.5]].")
def example_lu(sub):
    perm, l, u = sub.lu_decompose(A)
    close(perm, [1, 0], atol=0, what="the permutation")
    close(l, [[1.0, 0.0], [0.5, 1.0]], atol=1e-12, what="L")
    close(u, [[4.0, 3.0], [0.0, -0.5]], atol=1e-12, what="U")


@check("examples", "solve a 2x2 system", "The system 2x + y = 3, 4x + 3y = 7 has the solution x = 1, y = 1.")
def example_solve(sub):
    close(sub.lu_solve(A, np.array([3.0, 7.0])), [1.0, 1.0], atol=1e-12, what="the solution")


@check("examples", "forward and back substitution", "L y = b is solved from the top down, and U x = y from the bottom up.")
def example_triangular(sub):
    close(sub.forward_substitution(np.array([[2.0, 0.0], [1.0, 4.0]]), np.array([4.0, 10.0])), [2.0, 2.0], atol=1e-12, what="forward substitution")
    close(sub.back_substitution(np.array([[2.0, 1.0], [0.0, 4.0]]), np.array([5.0, 8.0])), [1.5, 2.0], atol=1e-12, what="back substitution")


@check("properties", "P A = L U", "Reordering the rows of A with the permutation gives the product of L and U.")
def factorization(sub):
    rng = np.random.default_rng(0)
    for n in (3, 5, 7):
        a = rng.normal(size=(n, n))
        perm, l, u = sub.lu_decompose(a)
        close(a[perm], l @ u, atol=1e-9, what=f"A[perm] and L U for n={n}")


@check("properties", "L is unit lower triangular with entries of size at most 1, U is upper triangular", "Partial pivoting divides by the largest entry in the column, so every multiplier has an absolute value of at most 1.")
def triangular_shape(sub):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(6, 6))
    perm, l, u = sub.lu_decompose(a)
    shape(l, (6, 6), what="L")
    close(np.diag(l), np.ones(6), atol=0, what="the diagonal of L")
    close(np.triu(l, 1), np.zeros((6, 6)), atol=0, what="the part of L above the diagonal")
    close(np.tril(u, -1), np.zeros((6, 6)), atol=0, what="the part of U below the diagonal")
    true(bool(np.all(np.abs(l) <= 1 + 1e-12)), "an entry of L is larger than 1", "all entries at most 1", np.abs(l).max())
    true(sorted(perm.tolist()) == list(range(6)), "perm is not a permutation", "each index once", perm)


@check("properties", "the solution matches numpy and has a small residual", "A x must equal b, and x must match the library solution.")
def solve_matches(sub):
    rng = np.random.default_rng(2)
    for n in (2, 4, 8):
        a = rng.normal(size=(n, n))
        b = rng.normal(size=n)
        x = sub.lu_solve(a, b)
        close(a @ x, b, atol=1e-9, what=f"A x for n={n}")
        close(x, np.linalg.solve(a, b), rtol=1e-8, atol=1e-9, what=f"the solution for n={n}")


@check("properties", "triangular solves match numpy", "Forward and back substitution give the same answer as a general solver on a triangular matrix.")
def triangular_matches(sub):
    rng = np.random.default_rng(3)
    m = rng.normal(size=(5, 5)) + 5 * np.eye(5)
    b = rng.normal(size=5)
    close(sub.forward_substitution(np.tril(m), b), np.linalg.solve(np.tril(m), b), rtol=1e-9, atol=1e-10, what="forward substitution")
    close(sub.back_substitution(np.triu(m), b), np.linalg.solve(np.triu(m), b), rtol=1e-9, atol=1e-10, what="back substitution")


@check("properties", "the input is not changed", "The factorization must work on a copy of the matrix.")
def input_unchanged(sub):
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    b = np.array([1.0, 2.0, 3.0])
    saved = (a.copy(), b.copy())
    sub.lu_decompose(a)
    sub.lu_solve(a, b)
    close(a, saved[0], atol=0, rtol=0, what="A")
    close(b, saved[1], atol=0, rtol=0, what="b")


@check("edge", "a zero in the corner needs a swap", "The entry a[0, 0] is 0, so the method must swap rows before it divides.")
def zero_corner(sub):
    a = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 0.0], [3.0, 0.0, 1.0]])
    perm, l, u = sub.lu_decompose(a)
    close(a[perm], l @ u, atol=1e-12, what="A[perm] and L U")
    close(sub.lu_solve(a, np.array([3.0, 2.0, 4.0])), np.linalg.solve(a, np.array([3.0, 2.0, 4.0])), rtol=1e-9, what="the solution")


@check("edge", "a swap after the first step", "The second pivot is in the third row. The multiplier that is already stored for the rows must swap with them.")
def late_swap(sub):
    a = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 1.0], [3.0, 7.0, 2.0]])
    perm, l, u = sub.lu_decompose(a)
    close(a[perm], l @ u, atol=1e-12, what="A[perm] and L U")


@check("edge", "a singular matrix raises ValueError", "If a column has no usable pivot, the matrix has no LU factorization with a non-zero diagonal.")
def singular(sub):
    raises(ValueError, sub.lu_decompose, np.array([[1.0, 2.0], [2.0, 4.0]]))
    raises(ValueError, sub.lu_solve, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]), np.ones(3))


@check("edge", "a 1x1 matrix", "The factorization of [[a]] is L = [[1]] and U = [[a]].")
def one_by_one(sub):
    perm, l, u = sub.lu_decompose(np.array([[4.0]]))
    close(l, [[1.0]], what="L")
    close(u, [[4.0]], what="U")
    close(sub.lu_solve(np.array([[4.0]]), np.array([2.0])), [0.5], what="the solution")


@fixture("systems")
def systems(ref):
    rng = np.random.default_rng(42)
    for n in (2, 3, 5, 8):
        a = rng.normal(size=(n, n))
        b = rng.normal(size=n)
        perm, l, u = ref.lu_decompose(a)
        yield {"a": a, "b": b}, {"perm": perm, "l": l, "u": u, "x": ref.lu_solve(a, b)}


@check("reference", "factors and solutions match the reference", "Your permutation, L, U and solution equal the reference on random systems.")
def reference_all(sub):
    for case, expected in fixtures("systems"):
        a, b = np.array(case["a"]), np.array(case["b"])
        perm, l, u = sub.lu_decompose(a)
        close(perm, expected["perm"], atol=0, what="the permutation")
        close(l, expected["l"], rtol=1e-8, atol=1e-10, what="L")
        close(u, expected["u"], rtol=1e-8, atol=1e-10, what="U")
        close(sub.lu_solve(a, b), expected["x"], rtol=1e-8, atol=1e-10, what="the solution")
