import numpy as np

from practice_check import check, close, fixture, fixtures, shape


@check("examples", "scale first, then shear", "The first argument acts first, so the product is shear times scaling.")
def example_scale_shear(sub):
    close(sub.compose(sub.scaling(2.0, 1.0), sub.shear(1.0, 0.0)), [[2.0, 1.0], [0.0, 1.0]], what="the composed matrix")


@check("examples", "shear first, then scale", "With the order reversed, the product is scaling times shear, and the result is different.")
def example_shear_scale(sub):
    close(sub.compose(sub.shear(1.0, 0.0), sub.scaling(2.0, 1.0)), [[2.0, 2.0], [0.0, 1.0]], what="the composed matrix")


@check("examples", "apply a matrix to two points", "Each row of the result is the matrix times the point in the same row of the input.")
def example_apply(sub):
    m = sub.compose(sub.scaling(2.0, 1.0), sub.shear(1.0, 0.0))
    close(sub.apply(m, np.array([[1.0, 0.0], [0.0, 1.0]])), [[2.0, 0.0], [1.0, 1.0]], what="the points")


@check("examples", "compose with no argument is the identity", "Doing no transformation leaves every point where it is.")
def example_empty(sub):
    close(sub.compose(), np.eye(2), what="the identity")


@check("properties", "rotation turns counter-clockwise", "A quarter turn sends the point (1, 0) to (0, 1).")
def rotation_direction(sub):
    close(sub.apply(sub.rotation(np.pi / 2), np.array([[1.0, 0.0]])), [[0.0, 1.0]], atol=1e-12, what="the rotated point")


@check("properties", "a rotation is orthogonal with determinant 1", "R^T R is the identity, and the determinant is 1, so lengths and areas do not change.")
def rotation_orthogonal(sub):
    for theta in (0.3, 1.0, -2.0):
        r = sub.rotation(theta)
        close(r.T @ r, np.eye(2), atol=1e-12, what="R^T R")
        close(np.linalg.det(r), 1.0, atol=1e-12, what="the determinant")


@check("properties", "two rotations add their angles", "Turning by a and then by b is the same as turning by a + b.")
def rotation_adds(sub):
    close(sub.compose(sub.rotation(0.4), sub.rotation(0.7)), sub.rotation(1.1), atol=1e-12, what="the composed rotation")


@check("properties", "the determinant of a composition is the product of determinants", "A scaling by 2 and 3 changes areas by 6. A rotation and a shear with ky = 0 change no area.")
def determinant_product(sub):
    m = sub.compose(sub.scaling(2.0, 3.0), sub.rotation(0.9), sub.shear(0.5, 0.0))
    close(np.linalg.det(m), 6.0, atol=1e-9, what="the determinant")


@check("properties", "the shear matrix has kx at the top right", "shear(kx, ky) is [[1, kx], [ky, 1]]. A point (0, 1) moves right by kx.")
def shear_layout(sub):
    close(sub.shear(2.0, 3.0), [[1.0, 2.0], [3.0, 1.0]], what="the shear matrix")


@check("properties", "apply matches a loop over the points", "Row i of the result must be the matrix times point i, as a column vector.")
def apply_matches_loop(sub):
    rng = np.random.default_rng(0)
    m = rng.normal(size=(2, 2))
    pts = rng.normal(size=(6, 2))
    shape(sub.apply(m, pts), (6, 2), what="the points")
    close(sub.apply(m, pts), np.array([m @ p for p in pts]), atol=1e-12, what="the points")


@check("properties", "the inputs are not changed", "A function that writes into its input arrays breaks the code that calls it.")
def inputs_unchanged(sub):
    a = np.array([[2.0, 1.0], [0.0, 1.0]])
    b = np.array([[0.0, -1.0], [1.0, 0.0]])
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    saved = (a.copy(), b.copy(), pts.copy())
    sub.compose(a, b)
    sub.apply(a, pts)
    for current, old in zip((a, b, pts), saved):
        close(current, old, atol=0, rtol=0, what="an input")


@check("edge", "one matrix composes to itself", "A product of a single matrix is the matrix.")
def single(sub):
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    close(sub.compose(m), m, what="the composed matrix")


@check("edge", "no points", "An input with 0 points gives an output with 0 points and the shape (0, 2).")
def no_points(sub):
    shape(sub.apply(np.eye(2), np.zeros((0, 2))), (0, 2), what="the points")


@check("edge", "scaling by zero collapses an axis", "A zero scale sends every point onto a line. The matrix has determinant 0.")
def collapse(sub):
    out = sub.apply(sub.scaling(0.0, 1.0), np.array([[5.0, 3.0], [-2.0, 4.0]]))
    close(out, [[0.0, 3.0], [0.0, 4.0]], what="the points")


@fixture("pipelines")
def pipelines(ref):
    rng = np.random.default_rng(42)
    for count in (1, 2, 3, 5):
        mats = [ref.rotation(rng.uniform(-3, 3)), ref.scaling(*rng.uniform(0.5, 2, size=2)), ref.shear(*rng.uniform(-1, 1, size=2)), ref.rotation(rng.uniform(-3, 3)), ref.scaling(*rng.uniform(0.5, 2, size=2))][:count]
        pts = rng.normal(size=(4, 2))
        yield {"mats": mats, "pts": pts}, {"matrix": ref.compose(*mats), "points": ref.apply(ref.compose(*mats), pts)}


@check("reference", "pipelines match the reference", "Your composed matrix and your transformed points equal the reference for pipelines of 1 to 5 steps.")
def reference_pipelines(sub):
    for case, expected in fixtures("pipelines"):
        mats = [np.array(m) for m in case["mats"]]
        pts = np.array(case["pts"])
        close(sub.compose(*mats), expected["matrix"], rtol=1e-9, atol=1e-12, what=f"the matrix of a pipeline with {len(mats)} steps")
        close(sub.apply(sub.compose(*mats), pts), expected["points"], rtol=1e-9, atol=1e-12, what=f"the points of a pipeline with {len(mats)} steps")
