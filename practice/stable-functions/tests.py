import numpy as np

from practice_check import check, close, fixture, fixtures, shape, true


def quiet(fn, *args, **kwargs):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return fn(*args, **kwargs)


@check("examples", "sigmoid at large inputs", "sigmoid(-1000) is 0 and sigmoid(1000) is 1. The code must not overflow on the way.")
def example_sigmoid(sub):
    close(quiet(sub.sigmoid, np.array([-1000.0, 0.0, 1000.0])), [0.0, 0.5, 1.0], atol=1e-12, what="sigmoid")


@check("examples", "softplus at large and small inputs", "softplus(1000) is 1000. softplus(-40) is about exp(-40) = 4.2e-18, and not 0.")
def example_softplus(sub):
    close(quiet(sub.softplus, np.array([-40.0, 0.0, 1000.0])), [4.248354255291589e-18, 0.6931471805599453, 1000.0], rtol=1e-9, what="softplus")


@check("examples", "log-sum-exp of two large equal numbers", "log(exp(1000) + exp(1000)) = 1000 + log 2. Subtract the maximum before exp.")
def example_lse(sub):
    close(quiet(sub.logsumexp, np.array([1000.0, 1000.0])), 1000.6931471805599, rtol=1e-12, what="logsumexp")


@check("examples", "log-softmax of large logits", "log_softmax(x) = x - logsumexp(x). For [1000, 1001] this is [-1.3132..., -0.3132...].")
def example_log_softmax(sub):
    close(quiet(sub.log_softmax, np.array([1000.0, 1001.0])), [-1.3132616875182228, -0.31326168751822286], rtol=1e-12, what="log_softmax")


@check("properties", "sigmoid(-x) = 1 - sigmoid(x)", "The sigmoid curve is symmetric around the point (0, 0.5).")
def sigmoid_symmetry(sub):
    x = np.linspace(-30, 30, 25)
    close(sub.sigmoid(-x), 1 - sub.sigmoid(x), atol=1e-12, what="sigmoid(-x)")


@check("properties", "sigmoid has the right values and shape", "The values are 1 / (1 + exp(-x)). The shape of the input is kept.")
def sigmoid_values(sub):
    x = np.array([[-2.0, 0.5], [3.0, -0.1]])
    out = sub.sigmoid(x)
    shape(out, (2, 2), what="sigmoid")
    close(out, 1 / (1 + np.exp(-x)), rtol=1e-12, what="sigmoid")


@check("properties", "softplus equals log(1 + exp(x)) where that is safe, and is above relu", "For moderate x, both formulas agree. softplus is always above max(0, x).")
def softplus_values(sub):
    x = np.linspace(-10, 10, 21)
    close(sub.softplus(x), np.log1p(np.exp(x)), rtol=1e-12, what="softplus")
    true(bool(np.all(sub.softplus(x) >= np.maximum(x, 0))), "softplus is below relu", "at least max(0, x)", sub.softplus(x))


@check("properties", "logsumexp uses the axis argument", "The reduction runs along the given axis, and that axis disappears.")
def lse_axis(sub):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4, 5))
    for axis in (0, 1, 2, -1):
        expected = np.log(np.sum(np.exp(x), axis=axis))
        out = sub.logsumexp(x, axis=axis)
        shape(out, expected.shape, what=f"the result for axis {axis}")
        close(out, expected, rtol=1e-10, what=f"the result for axis {axis}")


@check("properties", "log_softmax rows are log-probabilities", "The exp of each row sums to 1. Adding a constant to a row does not change the result.")
def log_softmax_props(sub):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 6)) * 5
    out = sub.log_softmax(x)
    close(np.exp(out).sum(axis=1), np.ones(4), atol=1e-12, what="the row sums of exp(log_softmax)")
    close(sub.log_softmax(x + 123.0), out, atol=1e-9, what="log_softmax of the shifted input")


@check("properties", "the input is not changed", "A function that writes into its input arrays breaks the code that calls it.")
def input_unchanged(sub):
    x = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 0.0]])
    saved = x.copy()
    sub.sigmoid(x)
    sub.softplus(x)
    sub.logsumexp(x)
    sub.log_softmax(x)
    close(x, saved, atol=0, rtol=0, what="x")


@check("edge", "a row of minus infinity", "If every value is minus infinity, the sum of exponentials is 0 and the log is minus infinity. The result must not be NaN.")
def all_minus_inf(sub):
    with np.errstate(all="ignore"):
        out = sub.logsumexp(np.array([[-np.inf, -np.inf], [0.0, 0.0]]), axis=1)
    true(out[0] == -np.inf, "the first row must give minus infinity", -np.inf, out[0])
    close(out[1], np.log(2.0), rtol=1e-12, what="the second row")


@check("edge", "one entry is minus infinity", "A value of minus infinity adds 0 to the sum. The other values give the result.")
def one_minus_inf(sub):
    close(sub.logsumexp(np.array([-np.inf, 1.0, 2.0])), np.log(np.exp(1.0) + np.exp(2.0)), rtol=1e-12, what="logsumexp")


@check("edge", "log_softmax of a very unequal row stays finite", "The log-probability of the small entry is about -2000. The log of a softmax that rounded to 0 would be minus infinity.")
def log_softmax_extreme(sub):
    close(quiet(sub.log_softmax, np.array([0.0, 2000.0])), [-2000.0, 0.0], atol=1e-9, what="log_softmax")


@fixture("data")
def data(ref):
    rng = np.random.default_rng(42)
    for shape_, scale in [((5,), 1.0), ((3, 4), 10.0), ((2, 3, 4), 200.0), ((6, 6), 800.0)]:
        x = rng.normal(size=shape_) * scale
        yield {"x": x}, {"sig": ref.sigmoid(x), "sp": ref.softplus(x), "lse": ref.logsumexp(x, axis=-1), "ls": ref.log_softmax(x, axis=-1), "lse0": ref.logsumexp(x, axis=0)}


@check("reference", "all four functions match the reference", "Your functions agree with the reference on tensors with values of size up to 2000.")
def reference_all(sub):
    for case, expected in fixtures("data"):
        x = np.array(case["x"])
        close(sub.sigmoid(x), expected["sig"], rtol=1e-9, atol=1e-300, what="sigmoid")
        close(sub.softplus(x), expected["sp"], rtol=1e-9, atol=1e-300, what="softplus")
        close(sub.logsumexp(x, axis=-1), expected["lse"], rtol=1e-9, what="logsumexp")
        close(sub.logsumexp(x, axis=0), expected["lse0"], rtol=1e-9, what="logsumexp over axis 0")
        close(sub.log_softmax(x, axis=-1), expected["ls"], rtol=1e-9, what="log_softmax")
