import numpy as np

from practice_check import check, close, equal, fixture, fixtures, numeric_grad, shape, true


@check("examples", "the loss at the minimum is 0", "At (1, 1) both terms of the Rosenbrock function are 0.")
def example_minimum(sub):
    close(sub.rosenbrock(np.array([1.0, 1.0])), 0.0, atol=1e-12, what="the loss at (1, 1)")


@check("examples", "the gradient at (-1, 1)", "At (-1, 1): df/dx = -2 * 2 - 400 * (-1) * 0 = -4 and df/dy = 200 * 0 = 0.")
def example_grad(sub):
    close(sub.rosenbrock_grad(np.array([-1.0, 1.0])), [-4.0, 0.0], atol=1e-12, what="the gradient")


@check("examples", "100 steps from (-1, 1)", "With learning rate 0.001, the loss goes from 4.0 to about 3.66 in 100 steps, and the history has 101 entries.")
def example_descent(sub):
    point, history = sub.gradient_descent(np.array([-1.0, 1.0]), 0.001, 100)
    close(point, [-0.91265103, 0.84098998], rtol=1e-6, what="the final point")
    close(history[0], 4.0, what="the first history entry")
    close(history[-1], 3.664727, rtol=1e-6, what="the last history entry")
    equal(len(history), 101, what="the length of the history")


@check("properties", "the gradient matches finite differences", "A gradient is correct when a small change in one coordinate changes the loss by gradient times that change.")
def finite_differences(sub):
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = rng.normal(size=2)
        close(sub.rosenbrock_grad(p), numeric_grad(sub.rosenbrock, p), atol=1e-4, rtol=1e-5, what=f"the gradient at {p}")


@check("properties", "the gradient is 0 at the minimum", "At a minimum of a smooth function, the gradient is 0.")
def zero_at_minimum(sub):
    close(sub.rosenbrock_grad(np.array([1.0, 1.0])), [0.0, 0.0], atol=1e-12, what="the gradient at (1, 1)")


@check("properties", "the loss is never negative", "The function is a sum of two squares, so it is 0 or larger everywhere.")
def non_negative(sub):
    rng = np.random.default_rng(1)
    for p in rng.normal(size=(20, 2)) * 2:
        true(sub.rosenbrock(p) >= 0.0, "the loss is negative", "0 or more", sub.rosenbrock(p))


@check("properties", "a small learning rate lowers the loss at every step", "A step against the gradient with a small enough rate cannot raise the loss.")
def monotone(sub):
    _, history = sub.gradient_descent(np.array([-1.2, 1.0]), 1e-4, 50)
    true(all(b <= a + 1e-12 for a, b in zip(history, history[1:])), "the loss went up in one step", "a non-increasing history", history[:5])


@check("properties", "the history starts with the loss of the start point", "The first entry is the loss before any update.")
def history_start(sub):
    start = np.array([0.3, -0.7])
    _, history = sub.gradient_descent(start, 0.001, 10)
    close(history[0], sub.rosenbrock(start), what="the first history entry")
    close(history[-1], sub.rosenbrock(sub.gradient_descent(start, 0.001, 10)[0]), what="the last history entry")


@check("properties", "the start point is not changed", "Write the update into a new array. The caller's start point must stay as it was.")
def start_unchanged(sub):
    start = np.array([-1.0, 1.0])
    sub.gradient_descent(start, 0.001, 20)
    close(start, [-1.0, 1.0], atol=0, rtol=0, what="the start point")


@check("edge", "zero steps", "With 0 steps the point does not move and the history has one entry.")
def zero_steps(sub):
    point, history = sub.gradient_descent(np.array([0.5, 0.5]), 0.01, 0)
    close(point, [0.5, 0.5], atol=0, rtol=0, what="the point")
    equal(len(history), 1, what="the length of the history")


@check("edge", "a large learning rate diverges", "At (-1, 1) with learning rate 0.1, the steps are too large. After 3 steps the loss is far above the start value.")
def divergence(sub):
    _, history = sub.gradient_descent(np.array([-1.0, 1.0]), 0.1, 3)
    true(history[-1] > 1000 * history[0], "the loss did not grow", "a loss far above the start value", history[-1])


@check("edge", "starting at the minimum stays there", "The gradient is 0 at (1, 1), so the point does not move.")
def stay_at_minimum(sub):
    point, history = sub.gradient_descent(np.array([1.0, 1.0]), 0.01, 25)
    close(point, [1.0, 1.0], atol=1e-12, what="the point")
    close(history[-1], 0.0, atol=1e-12, what="the last loss")


@fixture("runs")
def runs(ref):
    for start, lr, steps in [([-1.0, 1.0], 0.001, 100), ([0.0, 0.0], 0.002, 50), ([2.0, 2.0], 0.0005, 30), ([-1.5, 2.0], 0.001, 200)]:
        point, history = ref.gradient_descent(np.array(start), lr, steps)
        yield {"start": start, "lr": lr, "steps": steps}, {"point": point, "history": history, "grad": ref.rosenbrock_grad(np.array(start))}


@check("reference", "descent runs match the reference", "Your final point, your history and your gradient equal the reference for four runs.")
def reference_runs(sub):
    for case, expected in fixtures("runs"):
        point, history = sub.gradient_descent(np.array(case["start"]), case["lr"], case["steps"])
        close(point, expected["point"], rtol=1e-9, atol=1e-12, what=f"the final point for start {case['start']}")
        close(history, expected["history"], rtol=1e-9, atol=1e-12, what=f"the history for start {case['start']}")
        close(sub.rosenbrock_grad(np.array(case["start"])), expected["grad"], rtol=1e-9, atol=1e-12, what="the gradient")
