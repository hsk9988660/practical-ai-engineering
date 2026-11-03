import numpy as np

from practice_check import check, close, fixture, fixtures, raises, true

P2 = np.array([[0.9, 0.1], [0.5, 0.5]])
P3 = np.array([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]])


@check("examples", "one step from state 0", "The distribution after one step is the start distribution times the transition matrix. From state 0 it is the first row.")
def example_step(sub):
    close(sub.n_step(np.array([1.0, 0.0]), P2, 1), [0.9, 0.1], atol=1e-12, what="the distribution")


@check("examples", "two steps from state 0", "0.9 * 0.9 + 0.1 * 0.5 = 0.86 for state 0, and 0.14 for state 1.")
def example_two_steps(sub):
    close(sub.n_step(np.array([1.0, 0.0]), P2, 2), [0.86, 0.14], atol=1e-12, what="the distribution")


@check("examples", "the stationary distribution of a two-state chain", "The long-run share of time in state 0 is 5/6. It solves pi P = pi.")
def example_stationary(sub):
    close(sub.stationary_distribution(P2), [5 / 6, 1 / 6], rtol=1e-12, what="the stationary distribution")


@check("examples", "a simulated path", "From state 0 the cumulative probabilities are [0.9, 1.0]. u = 0.95 goes to state 1. From state 1 they are [0.5, 1.0], so u = 0.3 goes to 0, and then u = 0.6 stays at 0.")
def example_simulate(sub):
    true(list(sub.simulate(P2, 0, [0.95, 0.3, 0.6])) == [0, 1, 0, 0], "the path is wrong", [0, 1, 0, 0], list(sub.simulate(P2, 0, [0.95, 0.3, 0.6])))


@check("properties", "the stationary distribution satisfies pi P = pi", "A stationary distribution does not change after one step. It also adds up to 1 and has no negative entry.")
def stationary_fixed_point(sub):
    rng = np.random.default_rng(0)
    for k in (2, 4, 7):
        p = rng.random((k, k)) + 0.05
        p /= p.sum(axis=1, keepdims=True)
        pi = sub.stationary_distribution(p)
        close(pi @ p, pi, atol=1e-10, what=f"pi P for k={k}")
        close(pi.sum(), 1.0, atol=1e-12, what="the sum")
        true(bool(np.all(pi >= -1e-12)), "an entry is negative", "non-negative entries", pi)


@check("properties", "a known three-state chain", "This chain has the stationary distribution [0.25, 0.5, 0.25].")
def known_chain(sub):
    close(sub.stationary_distribution(P3), [0.25, 0.5, 0.25], atol=1e-12, what="the stationary distribution")


@check("properties", "a doubly stochastic matrix has the uniform distribution", "If the columns also add up to 1, the uniform distribution is stationary.")
def doubly_stochastic(sub):
    p = np.array([[0.2, 0.3, 0.5], [0.5, 0.2, 0.3], [0.3, 0.5, 0.2]])
    close(sub.stationary_distribution(p), np.full(3, 1 / 3), atol=1e-12, what="the stationary distribution")


@check("properties", "the distribution keeps its sum and converges to the stationary distribution", "After many steps, the start state is forgotten for a chain that mixes.")
def convergence(sub):
    pi = sub.stationary_distribution(P2)
    for start in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.3, 0.7])):
        out = sub.n_step(start, P2, 100)
        close(out.sum(), 1.0, atol=1e-12, what="the sum")
        close(out, pi, atol=1e-9, what="the distribution after 100 steps")


@check("properties", "n steps equal n times one step", "Taking 3 steps and then 4 steps is the same as taking 7 steps.")
def composition(sub):
    p0 = np.array([0.2, 0.3, 0.5])
    close(sub.n_step(sub.n_step(p0, P3, 3), P3, 4), sub.n_step(p0, P3, 7), atol=1e-12, what="the distribution")


@check("properties", "long simulations visit states with the stationary frequencies", "Over a long path, the share of time in each state is about the stationary probability.")
def simulation_frequency(sub):
    rng = np.random.default_rng(1)
    path = sub.simulate(P3, 0, rng.random(60000))
    close(np.bincount(path, minlength=3) / len(path), [0.25, 0.5, 0.25], atol=0.02, what="the state frequencies")


@check("properties", "the inputs are not changed", "A function that writes into its input arrays breaks the code that calls it.")
def inputs_unchanged(sub):
    p0, p = np.array([0.5, 0.5]), P2.copy()
    saved = (p0.copy(), p.copy())
    sub.n_step(p0, p, 3)
    sub.stationary_distribution(p)
    sub.simulate(p, 0, [0.1, 0.9])
    close(p0, saved[0], atol=0, rtol=0, what="the start distribution")
    close(p, saved[1], atol=0, rtol=0, what="the matrix")


@check("edge", "zero steps and no uniforms", "With 0 steps the distribution does not change. With no uniforms the path is the start state only.")
def zero_steps(sub):
    close(sub.n_step(np.array([0.3, 0.7]), P2, 0), [0.3, 0.7], atol=0, what="the distribution")
    true(list(sub.simulate(P2, 1, [])) == [1], "the path must be [1]", [1], list(sub.simulate(P2, 1, [])))


@check("edge", "an absorbing state keeps the path", "A state with probability 1 of staying never leaves, for every u.")
def absorbing(sub):
    p = np.array([[1.0, 0.0], [0.4, 0.6]])
    true(list(sub.simulate(p, 0, [0.0, 0.5, 0.99])) == [0, 0, 0, 0], "the path left the absorbing state", [0, 0, 0, 0], list(sub.simulate(p, 0, [0.0, 0.5, 0.99])))
    close(sub.n_step(np.array([0.0, 1.0]), p, 50), [1.0, 0.0], atol=1e-9, what="the distribution after 50 steps")


@check("edge", "a boundary value of u goes to the next state", "A u that equals a cumulative value belongs to the next state, as in the sampling exercise.")
def boundary(sub):
    true(list(sub.simulate(np.array([[0.5, 0.5], [0.5, 0.5]]), 0, [0.5])) == [0, 1], "u = 0.5 must go to state 1", [0, 1], list(sub.simulate(np.array([[0.5, 0.5], [0.5, 0.5]]), 0, [0.5])))


@check("edge", "a matrix that is not stochastic raises ValueError", "Each row must add up to 1 and have no negative entry.")
def invalid_matrix(sub):
    bad = np.array([[0.5, 0.6], [0.5, 0.5]])
    neg = np.array([[1.5, -0.5], [0.5, 0.5]])
    raises(ValueError, sub.n_step, np.array([1.0, 0.0]), bad, 1)
    raises(ValueError, sub.stationary_distribution, bad)
    raises(ValueError, sub.simulate, neg, 0, [0.1])


@fixture("chains")
def chains(ref):
    rng = np.random.default_rng(42)
    for k in (2, 3, 5, 6):
        p = rng.random((k, k)) + 0.1
        p /= p.sum(axis=1, keepdims=True)
        p0 = rng.random(k)
        p0 /= p0.sum()
        u = rng.random(30)
        yield {"p": p, "p0": p0, "u": u}, {"step": ref.n_step(p0, p, 7), "pi": ref.stationary_distribution(p), "path": [int(s) for s in ref.simulate(p, 0, u)]}


@check("reference", "all three functions match the reference", "Your n-step distribution, stationary distribution and simulated path equal the reference on random chains.")
def reference_all(sub):
    for case, expected in fixtures("chains"):
        p, p0, u = np.array(case["p"]), np.array(case["p0"]), list(case["u"])
        close(sub.n_step(p0, p, 7), expected["step"], rtol=1e-9, atol=1e-12, what="the distribution after 7 steps")
        close(sub.stationary_distribution(p), expected["pi"], rtol=1e-8, atol=1e-10, what="the stationary distribution")
        true([int(s) for s in sub.simulate(p, 0, u)] == expected["path"], "the path differs from the reference", expected["path"], list(sub.simulate(p, 0, u)))
