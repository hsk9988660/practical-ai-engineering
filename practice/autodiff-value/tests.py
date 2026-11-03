import math

from practice_check import check, close, equal, fixture, fixtures, true


def numeric_partial(expr, args, index, h=1e-6):
    plus = list(args)
    minus = list(args)
    plus[index] += h
    minus[index] -= h
    return (expr(*[Value_(a) for a in plus]).data - expr(*[Value_(a) for a in minus]).data) / (2 * h)


Value_ = None


def use(sub):
    global Value_
    Value_ = sub.Value
    return sub.Value


def mixed(a, b):
    return ((a * b + b.exp()).tanh() / (1 + a ** 2)) - 2 * b + (3 - a).relu()


@check("examples", "x*x + 3x at x = 2", "The value is 4 + 6 = 10. The derivative 2x + 3 is 7. The value x appears in two places, so two gradients add up.")
def example_poly(sub):
    Value = use(sub)
    x = Value(2.0)
    y = x * x + x * 3
    y.backward()
    close(y.data, 10.0, what="the value")
    close(x.grad, 7.0, what="the gradient of x")


@check("examples", "a tanh chain", "The chain rule multiplies the local derivatives of the square, the subtraction, the scaling and tanh.")
def example_tanh(sub):
    Value = use(sub)
    p = Value(0.5)
    q = (p.tanh() * 2 - 1) ** 2
    q.backward()
    close(q.data, 0.005740439096251332, rtol=1e-9, what="the value")
    close(p.grad, -0.23834300632936067, rtol=1e-9, what="the gradient of p")


@check("properties", "a new value has gradient 0 and float data", "The gradient starts at 0.0. An int input becomes a float.")
def initial_state(sub):
    Value = use(sub)
    v = Value(3)
    equal(v.data, 3.0, what="data")
    true(isinstance(v.data, float), "data must be a float", "float", type(v.data).__name__)
    equal(v.grad, 0.0, what="grad")


@check("properties", "the output gradient is 1", "backward() starts from d output / d output = 1. A leaf that is its own output has gradient 1.")
def root_gradient(sub):
    Value = use(sub)
    x = Value(5.0)
    x.backward()
    close(x.grad, 1.0, what="the gradient of a leaf output")
    a, b = Value(2.0), Value(3.0)
    c = a * b
    c.backward()
    close(c.grad, 1.0, what="the gradient of the output")


@check("properties", "gradients add up for a value that is used twice", "In a + a the gradient of a is 2. In a * a it is 2a.")
def accumulate(sub):
    Value = use(sub)
    a = Value(3.0)
    (a + a).backward()
    close(a.grad, 2.0, what="the gradient of a in a + a")
    b = Value(3.0)
    (b * b).backward()
    close(b.grad, 6.0, what="the gradient of b in b * b")


@check("properties", "a diamond graph", "z = (2x)(3x) = 6x^2, so dz/dx = 12x. Both branches pass their gradient back to x.")
def diamond(sub):
    Value = use(sub)
    x = Value(1.5)
    z = (x * 2) * (x * 3)
    z.backward()
    close(z.data, 13.5, what="the value")
    close(x.grad, 18.0, what="the gradient of x")


@check("properties", "a long chain needs the right order", "Multiplying by 1.01 fifty times gives a gradient of 1.01 to the power 50. A wrong visit order loses most of it.")
def long_chain(sub):
    Value = use(sub)
    x = Value(1.0)
    y = x
    for _ in range(50):
        y = y * 1.01
    y.backward()
    close(x.grad, 1.01 ** 50, rtol=1e-9, what="the gradient of x")


@check("properties", "the gradients match finite differences", "For a mix of operations, the gradient of each input equals the central difference of the output.")
def finite_differences(sub):
    Value = use(sub)
    args = [0.7, -0.4]
    a, b = Value(args[0]), Value(args[1])
    out = mixed(a, b)
    out.backward()
    close(a.grad, numeric_partial(mixed, args, 0), atol=1e-6, what="the gradient of a")
    close(b.grad, numeric_partial(mixed, args, 1), atol=1e-6, what="the gradient of b")


@check("properties", "plain numbers work on both sides", "2 * a, a + 1 and 3 - a must work. The derivatives are 2, 1 and -1.")
def reflected_ops(sub):
    Value = use(sub)
    a = Value(4.0)
    r = 2 * a + 1 + (3 - a)
    r.backward()
    close(r.data, 8.0, what="the value")
    close(a.grad, 1.0, what="the gradient of a")


@check("properties", "the derivative of a power", "d/dx x^3 is 3 x^2 and d/dx x^-1 is -x^-2.")
def powers(sub):
    Value = use(sub)
    x = Value(2.0)
    (x ** 3).backward()
    close(x.grad, 12.0, what="the gradient of x^3")
    y = Value(2.0)
    (y ** -1).backward()
    close(y.grad, -0.25, what="the gradient of x^-1")


@check("properties", "the derivative of division", "d/da (a / b) = 1 / b and d/db (a / b) = -a / b^2.")
def division(sub):
    Value = use(sub)
    a, b = Value(3.0), Value(4.0)
    (a / b).backward()
    close(a.grad, 0.25, what="the gradient of a")
    close(b.grad, -3.0 / 16.0, what="the gradient of b")


@check("edge", "relu has gradient 0 for negative input and at 0", "relu is flat for x <= 0. The code uses the convention that the derivative at 0 is 0.")
def relu_edges(sub):
    Value = use(sub)
    for x0, expected_data, expected_grad in [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 3.0, 1.0)]:
        x = Value(x0)
        y = x.relu()
        y.backward()
        close(y.data, expected_data, what=f"relu({x0})")
        close(x.grad, expected_grad, what=f"the gradient at {x0}")


@check("edge", "tanh and exp derivatives", "d tanh = 1 - tanh^2 and d exp = exp.")
def tanh_exp(sub):
    Value = use(sub)
    x = Value(0.8)
    x.tanh().backward()
    close(x.grad, 1 - math.tanh(0.8) ** 2, atol=1e-12, what="the gradient of tanh")
    y = Value(0.8)
    y.exp().backward()
    close(y.grad, math.exp(0.8), atol=1e-12, what="the gradient of exp")


@fixture("points")
def points(ref):
    for a0, b0 in [(0.7, -0.4), (-1.3, 0.2), (2.0, 1.1), (0.0, 0.5), (-0.5, -2.0)]:
        a, b = ref.Value(a0), ref.Value(b0)
        out = mixed(a, b)
        out.backward()
        yield {"a": a0, "b": b0}, {"data": out.data, "ga": a.grad, "gb": b.grad}


@check("reference", "value and gradients match the reference", "Your forward value and both gradients equal the reference for five points.")
def reference_points(sub):
    Value = use(sub)
    for case, expected in fixtures("points"):
        a, b = Value(case["a"]), Value(case["b"])
        out = mixed(a, b)
        out.backward()
        close(out.data, expected["data"], rtol=1e-9, atol=1e-12, what=f"the value at {case}")
        close(a.grad, expected["ga"], rtol=1e-9, atol=1e-12, what=f"the gradient of a at {case}")
        close(b.grad, expected["gb"], rtol=1e-9, atol=1e-12, what=f"the gradient of b at {case}")
