import numpy as np

from practice_check import check, close, fixture, fixtures, shape, true


def on_simplex(x, z=1.0):
    return bool(np.all(x >= -1e-12)) and abs(float(np.sum(x)) - z) < 1e-9


@check("examples", "a vector that is outside the simplex", "[2, 0] is not on the simplex. The nearest point is [1, 0].")
def example_corner(sub):
    close(sub.project_simplex(np.array([2.0, 0.0])), [1.0, 0.0], atol=1e-12, what="the projection")


@check("examples", "equal entries share the mass", "The point [1, 1, 1] is at the same distance from all corners, so the nearest point is the center.")
def example_center(sub):
    close(sub.project_simplex(np.array([1.0, 1.0, 1.0])), [1 / 3, 1 / 3, 1 / 3], atol=1e-12, what="the projection")


@check("examples", "a negative entry becomes 0", "The value -0.4 is clipped to 0, and the rest is shifted so that the sum is 1.")
def example_negative(sub):
    close(sub.project_simplex(np.array([0.8, 0.1, -0.4])), [0.85, 0.15, 0.0], atol=1e-12, what="the projection")


@check("examples", "projected gradient descent finds the nearest point", "Minimizing |x - c|^2 over the simplex gives the projection of c. For c = [2, 0, 0] this is [1, 0, 0].")
def example_pgd(sub):
    c = np.array([2.0, 0.0, 0.0])
    out = sub.projected_gradient(lambda x: 2 * (x - c), np.array([1 / 3, 1 / 3, 1 / 3]), sub.project_simplex, 0.25, 50)
    close(out, [1.0, 0.0, 0.0], atol=1e-9, what="the result")


@check("properties", "the result is on the simplex", "All entries are 0 or more, and they add up to z.")
def lands_on_simplex(sub):
    rng = np.random.default_rng(0)
    for n in (2, 5, 20, 200):
        v = rng.normal(size=n) * 3
        out = sub.project_simplex(v)
        shape(out, (n,), what="the projection")
        true(on_simplex(out), "the result is not on the simplex", "non-negative entries that sum to 1", (out.min(), out.sum()))


@check("properties", "a point on the simplex does not move", "The nearest point to a point of the set is the point itself, so projecting twice gives the same result.")
def idempotent(sub):
    rng = np.random.default_rng(1)
    p = rng.random(6)
    p /= p.sum()
    close(sub.project_simplex(p), p, atol=1e-12, what="the projection of a point on the simplex")
    v = rng.normal(size=6)
    once = sub.project_simplex(v)
    close(sub.project_simplex(once), once, atol=1e-12, what="the second projection")


@check("properties", "no point of the simplex is closer", "The projection is the nearest point of the simplex. Random points of the simplex cannot be closer.")
def is_nearest(sub):
    rng = np.random.default_rng(2)
    v = rng.normal(size=5) * 2
    best = np.linalg.norm(v - sub.project_simplex(v))
    for _ in range(300):
        q = rng.dirichlet(np.ones(5))
        true(np.linalg.norm(v - q) >= best - 1e-12, "a random simplex point is closer", "a distance of at least the projection distance", np.linalg.norm(v - q))


@check("properties", "the projection is a shift and a clip", "Every positive output entry equals the input entry minus the same number theta, and the other entries are 0.")
def shift_and_clip(sub):
    rng = np.random.default_rng(3)
    v = rng.normal(size=8)
    out = sub.project_simplex(v)
    positive = out > 0
    theta = v[positive] - out[positive]
    close(theta, np.full(theta.shape, theta[0]), atol=1e-12, what="the shifts of the positive entries")
    true(bool(np.all(v[~positive] <= theta[0] + 1e-12)), "an entry that became 0 was above theta", "entries at most theta", v[~positive])


@check("properties", "the order of the entries does not matter", "Permuting the input permutes the output in the same way.")
def permutation(sub):
    rng = np.random.default_rng(4)
    v = rng.normal(size=7)
    order = rng.permutation(7)
    close(sub.project_simplex(v)[order], sub.project_simplex(v[order]), atol=1e-12, what="the permuted projection")


@check("properties", "the sum parameter z", "With z = 2 the entries add up to 2, and the shape of the solution is the same.")
def sum_parameter(sub):
    rng = np.random.default_rng(5)
    out = sub.project_simplex(rng.normal(size=6), z=2.0)
    true(on_simplex(out, z=2.0), "the result does not sum to 2", "non-negative entries that sum to 2", (out.min(), out.sum()))


@check("properties", "the input is not changed", "A function that writes into its input arrays breaks the code that calls it.")
def input_unchanged(sub):
    v = np.array([3.0, -1.0, 0.5, 2.0])
    saved = v.copy()
    sub.project_simplex(v)
    close(v, saved, atol=0, rtol=0, what="v")


@check("edge", "a single entry", "The only point of the simplex with one coordinate is [z].")
def single(sub):
    close(sub.project_simplex(np.array([-5.0])), [1.0], atol=1e-12, what="the projection")
    close(sub.project_simplex(np.array([7.0]), z=3.0), [3.0], atol=1e-12, what="the projection with z = 3")


@check("edge", "all entries are very negative or very large", "A shift of the whole vector does not change the projection.")
def shifted(sub):
    v = np.array([0.3, -0.2, 0.9, 0.1])
    close(sub.project_simplex(v + 1000.0), sub.project_simplex(v), atol=1e-9, what="the projection of the shifted vector")
    close(sub.project_simplex(v - 1000.0), sub.project_simplex(v), atol=1e-9, what="the projection of the vector shifted down")


@check("edge", "projected gradient descent projects the start point", "With 0 steps the result must still be on the simplex.")
def start_projected(sub):
    out = sub.projected_gradient(lambda x: x, np.array([5.0, 5.0]), sub.project_simplex, 0.1, 0)
    true(on_simplex(out), "the start point was not projected", "a point on the simplex", out)


@check("edge", "projected gradient descent stays on the simplex", "Every iterate is projected, so the result is feasible even for a large step.")
def pgd_feasible(sub):
    c = np.array([0.2, -1.0, 3.0])
    out = sub.projected_gradient(lambda x: 2 * (x - c), np.array([0.5, 0.25, 0.25]), sub.project_simplex, 0.9, 5)
    true(on_simplex(out), "the result is not on the simplex", "a point on the simplex", out)


@fixture("vectors")
def vectors(ref):
    rng = np.random.default_rng(42)
    for n, scale, z in [(2, 1.0, 1.0), (4, 2.0, 1.0), (10, 5.0, 2.5), (50, 1.0, 1.0)]:
        v = rng.normal(size=n) * scale
        yield {"v": v, "z": z}, ref.project_simplex(v, z)


@check("reference", "projections match the reference", "Your projection equals the reference for random vectors of several sizes and values of z.")
def reference_projection(sub):
    for case, expected in fixtures("vectors"):
        close(sub.project_simplex(np.array(case["v"]), case["z"]), expected, rtol=1e-9, atol=1e-12, what=f"the projection of a vector of size {len(case['v'])}")
