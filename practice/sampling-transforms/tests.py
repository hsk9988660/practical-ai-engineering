import numpy as np

from practice_check import check, close, fixture, fixtures, shape, true


@check("examples", "inverse CDF of the exponential distribution", "x = -log(1 - u) / rate. For u = 0.5 and rate 2 this is log(2) / 2.")
def example_exponential(sub):
    close(sub.exponential_from_uniform(np.array([0.0, 0.5, 0.9]), 2.0), [0.0, 0.34657359027997264, 1.151292546497023], rtol=1e-9, atol=1e-12, what="the samples")


@check("examples", "Box-Muller with u2 = 0", "With u2 = 0 the angle is 0, so z0 is the radius sqrt(-2 log u1) and z1 is 0.")
def example_box_muller(sub):
    z0, z1 = sub.box_muller(np.array([0.5]), np.array([0.0]))
    close(z0, [1.1774100225154747], rtol=1e-12, what="z0")
    close(z1, [0.0], atol=1e-12, what="z1")


@check("examples", "categorical sampling from uniforms", "Each u picks the first category whose cumulative probability is above u. A u that equals a boundary goes to the next category.")
def example_categorical(sub):
    out = sub.categorical_from_uniform(np.array([0.0, 0.3, 0.5, 0.99]), np.array([0.2, 0.3, 0.5]))
    close(out, [0, 1, 2, 2], atol=0, what="the categories")


@check("properties", "the exponential samples satisfy the CDF", "If x is the sample for u, then the exponential CDF at x must give u back: 1 - exp(-rate x) = u.")
def exponential_cdf(sub):
    u = np.linspace(0.01, 0.99, 25)
    x = sub.exponential_from_uniform(u, 3.0)
    close(1 - np.exp(-3.0 * x), u, atol=1e-12, what="the CDF of the samples")


@check("properties", "the exponential samples grow with u and have mean 1 / rate", "The transform is increasing. A fine grid of uniforms gives a mean of 1 / rate.")
def exponential_mean(sub):
    u = (np.arange(200000) + 0.5) / 200000
    x = sub.exponential_from_uniform(u, 4.0)
    true(bool(np.all(np.diff(x) > 0)), "the samples do not grow with u", "an increasing array", x[:3])
    close(np.mean(x), 0.25, rtol=2e-3, what="the mean of the samples")


@check("properties", "Box-Muller gives mean 0 and variance 1", "On a fine grid of uniforms, the pooled z0 and z1 values have mean 0 and variance 1.")
def box_muller_moments(sub):
    g = (np.arange(400) + 0.5) / 400
    u1, u2 = np.meshgrid(g, g)
    z0, z1 = sub.box_muller(u1.ravel(), u2.ravel())
    z = np.concatenate([z0, z1])
    close(np.mean(z), 0.0, atol=2e-3, what="the mean")
    close(np.var(z), 1.0, rtol=2e-2, what="the variance")


@check("properties", "Box-Muller radius: z0^2 + z1^2 = -2 log u1", "The pair (z0, z1) is a point in the plane with the radius sqrt(-2 log u1).")
def box_muller_radius(sub):
    rng = np.random.default_rng(0)
    u1, u2 = rng.uniform(0.01, 1, size=10), rng.uniform(size=10)
    z0, z1 = sub.box_muller(u1, u2)
    shape(z0, (10,), what="z0")
    close(z0 ** 2 + z1 ** 2, -2 * np.log(u1), rtol=1e-12, what="the squared radius")


@check("properties", "categorical frequencies follow the probabilities", "A fine grid of uniforms puts a share p_i of the points into category i.")
def categorical_frequencies(sub):
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    u = (np.arange(100000) + 0.5) / 100000
    out = sub.categorical_from_uniform(u, probs)
    close(np.bincount(out, minlength=4) / len(u), probs, atol=1e-3, what="the frequencies")


@check("properties", "categories with probability 0 are never chosen", "The cumulative probability does not grow over a zero-probability category, so no u falls into it.")
def zero_probability(sub):
    probs = np.array([0.0, 0.5, 0.0, 0.5])
    u = np.linspace(0.0, 0.9999, 1000)
    out = sub.categorical_from_uniform(u, probs)
    true(set(np.unique(out)) <= {1, 3}, "a category with probability 0 was chosen", "only 1 and 3", sorted(set(out)))


@check("edge", "u = 0 gives the first category with positive probability", "u = 0 is below the first positive cumulative value, so it must not select an earlier category with probability 0.")
def zero_u(sub):
    out = sub.categorical_from_uniform(np.array([0.0]), np.array([0.0, 0.7, 0.3]))
    close(out, [1], atol=0, what="the category")


@check("edge", "rounding in the cumulative sum does not give an invalid index", "Ten probabilities of 0.1 add up to 0.9999999999999999. A u at that value must still give the last category, 9.")
def last_category(sub):
    out = sub.categorical_from_uniform(np.array([0.9999999999999999]), np.full(10, 0.1))
    close(out, [9], atol=0, what="the category")


@check("edge", "u = 0 gives a sample of 0, and u near 1 stays finite", "The exponential sample for u = 0 is 0. For u very near 1 it is large but finite.")
def exponential_edges(sub):
    out = sub.exponential_from_uniform(np.array([0.0, 1 - 1e-16]), 1.0)
    close(out[0], 0.0, atol=0, what="the sample for u = 0")
    true(bool(np.isfinite(out[1])) and out[1] > 30, "the sample for u near 1 is wrong", "a large finite number", out[1])


@fixture("uniforms")
def uniforms(ref):
    rng = np.random.default_rng(42)
    for n in (5, 50, 200):
        u = rng.uniform(0.001, 0.999, size=n)
        v = rng.uniform(size=n)
        probs = rng.random(4)
        probs /= probs.sum()
        z0, z1 = ref.box_muller(u, v)
        yield {"u": u, "v": v, "probs": probs}, {"exp": ref.exponential_from_uniform(u, 1.7), "z0": z0, "z1": z1, "cat": ref.categorical_from_uniform(u, probs)}


@check("reference", "all three transforms match the reference", "Your outputs equal the reference outputs for the same uniform numbers.")
def reference_all(sub):
    for case, expected in fixtures("uniforms"):
        u, v, probs = np.array(case["u"]), np.array(case["v"]), np.array(case["probs"])
        close(sub.exponential_from_uniform(u, 1.7), expected["exp"], rtol=1e-10, atol=1e-12, what="the exponential samples")
        z0, z1 = sub.box_muller(u, v)
        close(z0, expected["z0"], rtol=1e-10, atol=1e-12, what="z0")
        close(z1, expected["z1"], rtol=1e-10, atol=1e-12, what="z1")
        close(sub.categorical_from_uniform(u, probs), expected["cat"], atol=0, what="the categories")
