import numpy as np

from practice_check import check, close, fixture, fixtures, shape, true

X = np.array([[1.0, 2.0], [2.0, 4.1], [3.0, 5.9], [4.0, 8.2]])


@check("examples", "the fit of a nearly linear data set", "The data lies almost on one line, so one component explains 99.97 percent of the variance. The mean is the column mean.")
def example_fit(sub):
    mean, components, ratio = sub.fit_pca(X, 1)
    close(mean, [2.5, 5.05], what="the mean")
    close(components, [[0.43958017, 0.89820336]], rtol=1e-6, what="the components")
    close(ratio, [0.99968563], rtol=1e-6, what="the ratio")


@check("examples", "the projection to one dimension", "The coordinate of each sample is its centered vector times the component.")
def example_transform(sub):
    mean, components, _ = sub.fit_pca(X, 1)
    close(sub.transform(X, mean, components).ravel(), [-3.3988905, -1.07308328, 0.98326294, 3.48871084], rtol=1e-6, what="the coordinates")


@check("examples", "the reconstruction from one component", "The reconstruction is z times the component plus the mean. It is close to the data, but not equal.")
def example_inverse(sub):
    mean, components, _ = sub.fit_pca(X, 1)
    z = sub.transform(X, mean, components)
    expected = [[1.00591515, 1.99710513], [2.02829388, 4.08615299], [2.93222289, 5.93317008], [4.03356809, 8.1835718]]
    close(sub.inverse_transform(z, mean, components), expected, rtol=1e-6, what="the reconstruction")


@check("properties", "the shapes are (d,), (k, d) and (k,)", "The components are the rows of a (k, d) array. The ratio has one number for each kept component.")
def shapes(sub):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 5))
    mean, components, ratio = sub.fit_pca(x, 3)
    shape(mean, (5,), what="the mean")
    shape(components, (3, 5), what="the components")
    shape(ratio, (3,), what="the ratio")
    shape(sub.transform(x, mean, components), (30, 3), what="the coordinates")


@check("properties", "the components are orthonormal", "The rows have length 1 and are at right angles to each other.")
def orthonormal(sub):
    rng = np.random.default_rng(1)
    _, components, _ = sub.fit_pca(rng.normal(size=(40, 6)), 4)
    close(components @ components.T, np.eye(4), atol=1e-9, what="W W^T")


@check("properties", "the ratios are in descending order and add up to 1 for all components", "The first component explains the most variance. With k = d the ratios sum to 1.")
def ratios(sub):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 4)) * np.array([5.0, 2.0, 1.0, 0.5])
    _, _, ratio = sub.fit_pca(x, 4)
    true(bool(np.all(np.diff(ratio) <= 1e-12)), "the ratios are not in descending order", "a non-increasing list", ratio)
    close(ratio.sum(), 1.0, atol=1e-12, what="the sum of all ratios")


@check("properties", "the coordinates of the training data have mean 0", "The data is centered before it is projected, so every coordinate has mean 0.")
def centered(sub):
    rng = np.random.default_rng(3)
    x = rng.normal(loc=10.0, size=(25, 3))
    mean, components, _ = sub.fit_pca(x, 2)
    close(sub.transform(x, mean, components).mean(axis=0), np.zeros(2), atol=1e-9, what="the mean of the coordinates")


@check("properties", "the variance of a coordinate is its eigenvalue", "The variance of the data along component j equals ratio j times the total variance.")
def variance_matches(sub):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(60, 3)) * np.array([3.0, 1.0, 0.3])
    mean, components, ratio = sub.fit_pca(x, 3)
    z = sub.transform(x, mean, components)
    total = np.var(x, axis=0, ddof=1).sum()
    close(np.var(z, axis=0, ddof=1), ratio * total, rtol=1e-9, what="the variance of each coordinate")


@check("properties", "all components give an exact reconstruction", "With k = d no information is dropped, so the data is rebuilt exactly.")
def full_reconstruction(sub):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 4))
    mean, components, _ = sub.fit_pca(x, 4)
    close(sub.inverse_transform(sub.transform(x, mean, components), mean, components), x, atol=1e-9, what="the reconstruction")


@check("properties", "the sign rule: the largest entry of each component is positive", "The sign of an eigenvector is arbitrary, so the rule fixes it. The entry with the largest absolute value must be positive.")
def sign_rule(sub):
    rng = np.random.default_rng(6)
    _, components, _ = sub.fit_pca(rng.normal(size=(30, 5)), 5)
    for row in components:
        true(row[np.argmax(np.abs(row))] > 0, "the largest entry of a component is negative", "a positive largest entry", row)


@check("properties", "the input is not changed", "A function that writes into its input arrays breaks the code that calls it.")
def input_unchanged(sub):
    x = np.array([[1.0, 2.0], [3.0, 1.0], [5.0, 4.0]])
    saved = x.copy()
    mean, components, _ = sub.fit_pca(x, 1)
    sub.transform(x, mean, components)
    close(x, saved, atol=0, rtol=0, what="x")


@check("edge", "data on a line has one component with ratio 1", "If all points lie on a line, one component explains all the variance and the others explain none.")
def collinear(sub):
    t = np.arange(5.0)
    x = np.column_stack([t, 2 * t])
    _, components, ratio = sub.fit_pca(x, 2)
    close(ratio, [1.0, 0.0], atol=1e-9, what="the ratio")
    close(np.abs(components[0]), [1 / np.sqrt(5), 2 / np.sqrt(5)], atol=1e-9, what="the first component")


@check("edge", "two samples", "Two samples give one non-zero direction: the line between them.")
def two_samples(sub):
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    _, components, ratio = sub.fit_pca(x, 1)
    close(components, [[0.6, 0.8]], atol=1e-9, what="the component")
    close(ratio, [1.0], atol=1e-9, what="the ratio")


@check("edge", "the ratio uses all eigenvalues, not only the kept ones", "With k = 1 of 3 components, the ratio is the share of the total, so it is below 1.")
def ratio_total(sub):
    x = np.diag([3.0, 2.0, 1.0])
    _, _, ratio = sub.fit_pca(x, 1)
    true(ratio[0] < 1.0, "the ratio is not below 1", "a share of the total variance", ratio)


@fixture("data")
def data(ref):
    rng = np.random.default_rng(42)
    for n, d, k in [(10, 2, 1), (30, 4, 2), (50, 6, 3), (20, 5, 5)]:
        x = rng.normal(size=(n, d)) @ rng.normal(size=(d, d))
        mean, components, ratio = ref.fit_pca(x, k)
        z = ref.transform(x, mean, components)
        yield {"x": x, "k": k}, {"mean": mean, "components": components, "ratio": ratio, "z": z, "back": ref.inverse_transform(z, mean, components)}


@check("reference", "fit, transform and inverse match the reference", "Your mean, components, ratio, coordinates and reconstruction equal the reference on random data.")
def reference_all(sub):
    for case, expected in fixtures("data"):
        x = np.array(case["x"])
        mean, components, ratio = sub.fit_pca(x, case["k"])
        close(mean, expected["mean"], rtol=1e-9, atol=1e-12, what="the mean")
        close(components, expected["components"], rtol=1e-6, atol=1e-8, what="the components")
        close(ratio, expected["ratio"], rtol=1e-8, atol=1e-10, what="the ratio")
        z = sub.transform(x, mean, components)
        close(z, expected["z"], rtol=1e-6, atol=1e-8, what="the coordinates")
        close(sub.inverse_transform(z, mean, components), expected["back"], rtol=1e-6, atol=1e-8, what="the reconstruction")
