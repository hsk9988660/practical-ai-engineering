import numpy as np

from practice_check import check, close, equal, fixture, fixtures, raises, shape, true


@check("examples", "log-density of the standard normal at 0", "The density at 0 is 1 / sqrt(2 pi), so the log is -0.5 log(2 pi).")
def example_pdf(sub):
    close(sub.gaussian_log_pdf(0.0, 0.0, 1.0), -0.9189385332046727, rtol=1e-12, what="the log-density")


@check("examples", "the maximum likelihood fit of 1, 2, 3, 4", "The mean is 2.5. The standard deviation divides by n, so it is sqrt(1.25).")
def example_fit(sub):
    mu, sigma = sub.fit_gaussian(np.array([1.0, 2.0, 3.0, 4.0]))
    close(mu, 2.5, what="mu")
    close(sigma, 1.118033988749895, rtol=1e-12, what="sigma")


@check("examples", "log-likelihood of the fitted data", "The log-likelihood is the sum of the four log-densities.")
def example_ll(sub):
    close(sub.log_likelihood(np.array([1.0, 2.0, 3.0, 4.0]), 2.5, 1.118033988749895), -6.12204123544711, rtol=1e-10, what="the log-likelihood")


@check("properties", "the log-density is symmetric around the mean", "The density depends on the distance to mu, not on the side.")
def symmetric(sub):
    close(sub.gaussian_log_pdf(3.0 + 1.7, 3.0, 2.0), sub.gaussian_log_pdf(3.0 - 1.7, 3.0, 2.0), atol=1e-12, what="the two sides")


@check("properties", "the density integrates to 1", "Take exp of the log-density on a fine grid. The area under the curve must be 1.")
def integrates(sub):
    grid = np.linspace(-20.0, 24.0, 4001)
    density = np.exp(sub.gaussian_log_pdf(grid, 2.0, 1.5))
    close(np.trapezoid(density, grid) if hasattr(np, "trapezoid") else np.trapz(density, grid), 1.0, rtol=1e-6, what="the area")


@check("properties", "an array input gives an array output", "The function works on every entry of an array and keeps the shape.")
def array_shape(sub):
    x = np.linspace(-2, 2, 7)
    out = sub.gaussian_log_pdf(x, 0.0, 1.0)
    shape(out, (7,), what="the log-densities")
    close(out, [sub.gaussian_log_pdf(v, 0.0, 1.0) for v in x], atol=1e-12, what="the log-densities")


@check("properties", "the fit maximizes the log-likelihood", "No other mean or standard deviation can give a larger log-likelihood than the maximum likelihood fit.")
def fit_is_maximum(sub):
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=200)
    mu, sigma = sub.fit_gaussian(x)
    best = sub.log_likelihood(x, mu, sigma)
    for dmu, dsigma in [(0.1, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1), (0.2, 0.2), (-0.3, 0.1)]:
        true(sub.log_likelihood(x, mu + dmu, sigma + dsigma) < best, "a different pair gave a larger log-likelihood", "less than the fitted value", (dmu, dsigma))


@check("properties", "the log-likelihood is a sum, not a mean", "The log of a product of independent densities is the sum of the logs. Doubling the data doubles the log-likelihood.")
def is_sum(sub):
    x = np.array([0.5, 1.5, -0.3])
    close(sub.log_likelihood(np.concatenate([x, x]), 0.2, 1.3), 2 * sub.log_likelihood(x, 0.2, 1.3), rtol=1e-12, what="the doubled log-likelihood")


@check("properties", "the input is not changed", "A function that writes into its input arrays breaks the code that calls it.")
def input_unchanged(sub):
    x = np.array([1.0, 2.0, 4.0])
    saved = x.copy()
    sub.fit_gaussian(x)
    sub.log_likelihood(x, 1.0, 1.0)
    sub.gaussian_log_pdf(x, 1.0, 1.0)
    close(x, saved, atol=0, rtol=0, what="x")


@check("edge", "a point far in the tail stays finite", "At 40 standard deviations the density is about 1e-348, below the smallest float. The log-density is -800.9189..., not minus infinity.")
def far_tail(sub):
    close(sub.gaussian_log_pdf(40.0, 0.0, 1.0), -800.9189385332047, rtol=1e-12, what="the log-density")


@check("edge", "sigma must be positive", "A standard deviation of 0 or below has no density. The function must raise ValueError.")
def bad_sigma(sub):
    raises(ValueError, sub.gaussian_log_pdf, 0.0, 0.0, 0.0)
    raises(ValueError, sub.gaussian_log_pdf, 0.0, 0.0, -1.0)


@check("edge", "data without spread fits sigma 0", "If all values are equal, the maximum likelihood standard deviation is 0.")
def constant_data(sub):
    mu, sigma = sub.fit_gaussian(np.array([5.0, 5.0, 5.0]))
    close(mu, 5.0, what="mu")
    close(sigma, 0.0, atol=1e-12, what="sigma")


@fixture("data")
def data(ref):
    rng = np.random.default_rng(42)
    for n in (3, 10, 50, 200):
        x = rng.normal(loc=rng.uniform(-5, 5), scale=rng.uniform(0.5, 4), size=n)
        mu, sigma = ref.fit_gaussian(x)
        yield {"x": x}, {"mu": mu, "sigma": sigma, "ll": ref.log_likelihood(x, mu + 0.3, sigma * 1.2), "pdf": ref.gaussian_log_pdf(x, mu, sigma)}


@check("reference", "fit, log-density and log-likelihood match the reference", "Your three functions agree with the reference on random data sets.")
def reference_all(sub):
    for case, expected in fixtures("data"):
        x = np.array(case["x"])
        mu, sigma = sub.fit_gaussian(x)
        close(mu, expected["mu"], rtol=1e-10, what="mu")
        close(sigma, expected["sigma"], rtol=1e-10, what="sigma")
        close(sub.gaussian_log_pdf(x, mu, sigma), expected["pdf"], rtol=1e-10, what="the log-densities")
        close(sub.log_likelihood(x, mu + 0.3, sigma * 1.2), expected["ll"], rtol=1e-10, what="the log-likelihood")
