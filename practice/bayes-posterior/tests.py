import numpy as np

from practice_check import check, close, fixture, fixtures, raises, shape


@check("examples", "a positive medical test", "With a prior of 1 percent, the posterior is 0.95 * 0.01 divided by the evidence 0.059, which is about 16 percent.")
def example_medical(sub):
    close(sub.posterior(np.array([0.01, 0.99]), np.array([0.95, 0.05])), [0.16101695, 0.83898305], rtol=1e-6, what="the posterior")


@check("examples", "three observations in a row", "Each observation multiplies the odds of hypothesis 0 against hypothesis 1 by 0.9 / 0.2 = 4.5.")
def example_sequential(sub):
    close(sub.sequential_posterior(np.array([0.5, 0.5]), np.array([[0.9, 0.2]] * 3)), [0.98914518, 0.01085482], rtol=1e-6, what="the posterior")


@check("properties", "the posterior adds up to 1", "The denominator rescales the products so that the result is a probability distribution.")
def sums_to_one(sub):
    rng = np.random.default_rng(0)
    prior = rng.random(5)
    prior /= prior.sum()
    out = sub.posterior(prior, rng.random(5))
    shape(out, (5,), what="the posterior")
    close(out.sum(), 1.0, atol=1e-12, what="the sum of the posterior")


@check("properties", "a flat likelihood leaves the prior unchanged", "If the data is equally likely under every hypothesis, it tells you nothing.")
def flat_likelihood(sub):
    prior = np.array([0.2, 0.3, 0.5])
    close(sub.posterior(prior, np.array([0.7, 0.7, 0.7])), prior, atol=1e-12, what="the posterior")


@check("properties", "a flat prior gives the normalized likelihood", "With equal priors, the posterior is proportional to the likelihood.")
def flat_prior(sub):
    out = sub.posterior(np.array([0.25, 0.25, 0.25, 0.25]), np.array([1.0, 2.0, 3.0, 4.0]))
    close(out, [0.1, 0.2, 0.3, 0.4], atol=1e-12, what="the posterior")


@check("properties", "the order of the observations does not matter", "Independent observations multiply, and multiplication is commutative.")
def order_free(sub):
    rng = np.random.default_rng(1)
    like = rng.random((6, 4)) + 0.1
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    close(sub.sequential_posterior(prior, like), sub.sequential_posterior(prior, like[::-1]), atol=1e-12, what="the posterior after reversing the order")


@check("properties", "sequential updates equal one update with the product", "For a few observations, one update with the product of all likelihood rows gives the same posterior.")
def equals_product(sub):
    rng = np.random.default_rng(2)
    like = rng.random((4, 3)) + 0.1
    prior = np.array([0.5, 0.3, 0.2])
    close(sub.sequential_posterior(prior, like), sub.posterior(prior, np.prod(like, axis=0)), atol=1e-12, what="the posterior")


@check("properties", "the inputs are not changed", "A function that writes into its input arrays breaks the code that calls it.")
def inputs_unchanged(sub):
    prior = np.array([0.4, 0.6])
    like = np.array([[0.2, 0.9], [0.5, 0.5]])
    saved = (prior.copy(), like.copy())
    sub.posterior(prior, like[0])
    sub.sequential_posterior(prior, like)
    close(prior, saved[0], atol=0, rtol=0, what="the prior")
    close(like, saved[1], atol=0, rtol=0, what="the likelihoods")


@check("edge", "a thousand observations do not underflow", "The product of 1000 likelihoods of 0.1 is 1e-1000, which a float cannot hold. Normalize after every observation.")
def many_observations(sub):
    like = np.tile(np.array([0.1, 0.05]), (1000, 1))
    out = sub.sequential_posterior(np.array([0.5, 0.5]), like)
    close(out, [1.0, 0.0], atol=1e-12, what="the posterior")


@check("edge", "a hypothesis with prior 0 stays at 0", "No data can raise a belief that started at exactly 0.")
def zero_prior(sub):
    out = sub.sequential_posterior(np.array([0.0, 1.0]), np.array([[0.9, 0.1], [0.9, 0.1]]))
    close(out, [0.0, 1.0], atol=1e-12, what="the posterior")


@check("edge", "zero evidence raises ValueError", "If every hypothesis gives the data probability 0, the posterior is not defined.")
def zero_evidence(sub):
    raises(ValueError, sub.posterior, np.array([0.5, 0.5]), np.array([0.0, 0.0]))


@check("edge", "no observations returns the prior", "With zero observations nothing changes.")
def no_observations(sub):
    out = sub.sequential_posterior(np.array([0.3, 0.7]), np.zeros((0, 2)))
    close(out, [0.3, 0.7], atol=1e-12, what="the posterior")


@fixture("cases")
def cases(ref):
    rng = np.random.default_rng(42)
    for t, h in [(1, 2), (3, 3), (10, 4), (50, 5)]:
        prior = rng.random(h) + 0.05
        prior /= prior.sum()
        like = rng.random((t, h)) + 0.05
        yield {"prior": prior, "like": like}, ref.sequential_posterior(prior, like)


@check("reference", "sequential posteriors match the reference", "Your posterior equals the reference for several numbers of observations and hypotheses.")
def reference_sequential(sub):
    for case, expected in fixtures("cases"):
        close(sub.sequential_posterior(case["prior"], case["like"]), expected, rtol=1e-9, atol=1e-12, what=f"the posterior for {len(case['like'])} observations")
