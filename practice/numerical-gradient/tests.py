import numpy as np

from practice_check import check, close, fixture, fixtures, shape, true


@check("examples", "gradient of x0^2 + 3 x0 x1", "The partial derivatives are 2 x0 + 3 x1 = 8 and 3 x0 = 3 at the point (1, 2).")
def example_quadratic(sub):
    f = lambda v: v[0] ** 2 + 3 * v[0] * v[1]
    close(sub.numerical_gradient(f, np.array([1.0, 2.0])), [8.0, 3.0], atol=1e-6, what="the gradient")


@check("examples", "relative error of two close gradients", "The difference has norm 0.001 and the norms sum to about 2, so the relative error is about 0.0005.")
def example_relative(sub):
    close(sub.relative_error(np.array([1.0, 0.0]), np.array([1.0, 0.001])), 0.00049999987, rtol=1e-5, what="the relative error")


@check("examples", "relative error of two zero arrays", "Both arrays are zero, so there is no difference and the answer is 0.0, not NaN.")
def example_zero(sub):
    close(sub.relative_error(np.zeros(3), np.zeros(3)), 0.0, atol=0, what="the relative error")


@check("properties", "the gradient of a linear function is its coefficient vector", "The derivative of a . x with respect to x is a, and the central difference is exact for a linear function.")
def linear_function(sub):
    a = np.array([2.0, -1.0, 0.5, 4.0])
    close(sub.numerical_gradient(lambda v: float(a @ v), np.array([0.3, 1.0, -2.0, 5.0])), a, atol=1e-7, what="the gradient")


@check("properties", "the gradient matches the analytic gradient of a smooth function", "For f = sum(sin(x) * x^2) the gradient is cos(x) x^2 + 2 x sin(x).")
def smooth_function(sub):
    x = np.array([0.5, -1.2, 2.0])
    analytic = np.cos(x) * x ** 2 + 2 * x * np.sin(x)
    close(sub.numerical_gradient(lambda v: float(np.sum(np.sin(v) * v ** 2)), x), analytic, atol=1e-6, what="the gradient")


@check("properties", "the result has the shape of the input", "The gradient has one entry for each entry of x, in the same layout, also for a matrix.")
def matrix_shape(sub):
    w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    grad = sub.numerical_gradient(lambda m: float(np.sum(m ** 2)), w)
    shape(grad, (2, 3), what="the gradient")
    close(grad, 2 * w, atol=1e-6, what="the gradient")


@check("properties", "a central difference is accurate to order h squared", "With h = 1e-3 the error for x^3 at x = 2 is about 1e-6 for a central difference, and about 1e-2 for a forward difference.")
def central_accuracy(sub):
    grad = sub.numerical_gradient(lambda v: float(v[0] ** 3), np.array([2.0]), h=1e-3)
    close(grad, [12.0], atol=1e-4, what="the gradient")


@check("properties", "the step size h is used", "A larger h is a rougher estimate. With h = 1 and f = x^3 at 0, the central difference is exactly 1.")
def step_used(sub):
    grad = sub.numerical_gradient(lambda v: float(v[0] ** 3), np.array([0.0]), h=1.0)
    close(grad, [1.0], atol=1e-12, what="the gradient")


@check("properties", "the input is not changed", "Write the steps into new arrays. The input must be exactly as it was.")
def input_unchanged(sub):
    x = np.array([0.1, 0.2, 0.3])
    saved = x.copy()
    sub.numerical_gradient(lambda v: float(np.sum(v ** 2)), x)
    close(x, saved, atol=0, rtol=0, what="x")


@check("properties", "relative error is symmetric and does not depend on scale", "Multiplying both arrays by a constant, or swapping them, does not change the relative error.")
def relative_properties(sub):
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.1, 1.9, 3.2])
    base = sub.relative_error(a, b)
    close(sub.relative_error(b, a), base, rtol=1e-12, what="the swapped error")
    close(sub.relative_error(1000 * a, 1000 * b), base, rtol=1e-12, what="the scaled error")
    close(sub.relative_error(a, a), 0.0, atol=0, what="the error of equal arrays")


@check("edge", "a one-entry input and a matrix input", "The code must visit every entry for any number of dimensions, including a single number in an array.")
def edge_shapes(sub):
    close(sub.numerical_gradient(lambda v: float(v[0] ** 2), np.array([3.0])), [6.0], atol=1e-6, what="the gradient of one entry")
    x = np.arange(6.0).reshape(2, 3)
    close(sub.numerical_gradient(lambda m: float(np.sum(m * np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))), x), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], atol=1e-7, what="the gradient of a matrix")


@check("edge", "one gradient is zero", "If b is zero and a is not, the relative error is exactly 1. The denominator is not zero.")
def one_zero(sub):
    close(sub.relative_error(np.array([3.0, 4.0]), np.zeros(2)), 1.0, atol=1e-12, what="the relative error")


@fixture("functions")
def functions(ref):
    rng = np.random.default_rng(42)
    for shape_ in [(3,), (5,), (2, 3), (4, 2)]:
        x = rng.normal(size=shape_)
        yield {"x": x}, ref.numerical_gradient(lambda v: float(np.sum(np.tanh(v) * v)), x)


@check("reference", "gradients match the reference", "Your estimate equals the reference estimate for several shapes.")
def reference_gradients(sub):
    for case, expected in fixtures("functions"):
        close(sub.numerical_gradient(lambda v: float(np.sum(np.tanh(v) * v)), case["x"]), expected, rtol=1e-6, atol=1e-8, what=f"the gradient for an input of shape {case['x'].shape}")
