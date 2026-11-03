import numpy as np

from practice_check import check, close, fixture, fixtures, shape, true

X = np.array([[1.0, 10.0, 5.0], [3.0, 20.0, 5.0], [5.0, 30.0, 5.0]])


@check("examples", "mean and scale of the example", "The mean is taken for each column. The scale is the population standard deviation, and it is 1.0 for the constant column.")
def example_fit(sub):
    mean, scale = sub.fit_standardizer(X)
    close(mean, [3.0, 20.0, 5.0], what="the mean")
    close(scale, [1.63299316, 8.16496581, 1.0], rtol=1e-7, what="the scale")


@check("examples", "the train data after standardizing", "Each column of the train data becomes (x - mean) / scale. The constant column becomes 0.")
def example_apply(sub):
    mean, scale = sub.fit_standardizer(X)
    expected = [[-1.22474487, -1.22474487, 0.0], [0.0, 0.0, 0.0], [1.22474487, 1.22474487, 0.0]]
    close(sub.apply_standardizer(X, mean, scale), expected, rtol=1e-7, atol=1e-12, what="the standardized data")


@check("examples", "new data uses the train statistics", "A new row is scaled with the mean and scale of the train data, not with its own numbers.")
def example_new_data(sub):
    mean, scale = sub.fit_standardizer(X)
    close(sub.apply_standardizer(np.array([[7.0, 25.0, 6.0]]), mean, scale), [[2.44948974, 0.61237244, 1.0]], rtol=1e-7, what="the standardized new row")


@check("properties", "mean and scale have one number for each column", "The statistics run down the rows, so each has the shape (n_features,).")
def stat_shapes(sub):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 5))
    mean, scale = sub.fit_standardizer(x)
    shape(mean, (5,), what="the mean")
    shape(scale, (5,), what="the scale")


@check("properties", "train columns have mean 0 and standard deviation 1", "After the standardizer is applied to its own train data, every column that is not constant has mean 0 and std 1.")
def zero_mean_unit_std(sub):
    rng = np.random.default_rng(1)
    x = rng.normal(loc=[5.0, -3.0, 100.0], scale=[1.0, 20.0, 0.01], size=(50, 3))
    mean, scale = sub.fit_standardizer(x)
    z = sub.apply_standardizer(x, mean, scale)
    close(np.mean(z, axis=0), np.zeros(3), atol=1e-9, what="the column means")
    close(np.std(z, axis=0), np.ones(3), atol=1e-9, what="the column standard deviations")


@check("properties", "test data is not standardized with its own statistics", "If the test data has another mean than the train data, its standardized mean is not 0. Do not fit again on the test data.")
def uses_train_stats(sub):
    rng = np.random.default_rng(2)
    train = rng.normal(size=(40, 2))
    test = train + np.array([10.0, -10.0])
    mean, scale = sub.fit_standardizer(train)
    z = sub.apply_standardizer(test, mean, scale)
    close(np.mean(z, axis=0), np.array([10.0, -10.0]) / scale, rtol=1e-9, what="the mean of the shifted test data after standardizing")


@check("properties", "the inputs are not changed", "A function that writes into its input arrays breaks the code that calls it.")
def inputs_unchanged(sub):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 3))
    saved = x.copy()
    mean, scale = sub.fit_standardizer(x)
    saved_stats = (mean.copy(), scale.copy())
    sub.apply_standardizer(x, mean, scale)
    close(x, saved, atol=0, rtol=0, what="x")
    close(mean, saved_stats[0], atol=0, rtol=0, what="mean")
    close(scale, saved_stats[1], atol=0, rtol=0, what="scale")


@check("properties", "the scale is the population standard deviation", "Divide by n and not by n - 1. For the two values 0 and 2, the population std is 1.")
def population_std(sub):
    _, scale = sub.fit_standardizer(np.array([[0.0], [2.0]]))
    close(scale, [1.0], what="the scale")


@check("edge", "a constant column gives 0 and not NaN", "A column with std 0 gets scale 1.0, so the standardized column is all zeros.")
def constant_column(sub):
    x = np.full((4, 2), 7.0)
    mean, scale = sub.fit_standardizer(x)
    close(scale, [1.0, 1.0], what="the scale")
    z = sub.apply_standardizer(x, mean, scale)
    true(bool(np.all(np.isfinite(z))), "the output has NaN or infinity", "all finite values", z)
    close(z, np.zeros((4, 2)), what="the standardized data")


@check("edge", "one row", "One row has std 0 in every column, so every scale is 1.0 and the row becomes zeros.")
def single_row(sub):
    x = np.array([[3.0, -4.0, 8.0]])
    mean, scale = sub.fit_standardizer(x)
    close(scale, [1.0, 1.0, 1.0], what="the scale")
    close(sub.apply_standardizer(x, mean, scale), [[0.0, 0.0, 0.0]], what="the standardized row")


@check("edge", "one column", "The code must work for a single feature. The shapes stay (1,) for the statistics.")
def single_column(sub):
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    mean, scale = sub.fit_standardizer(x)
    shape(mean, (1,), what="the mean")
    close(mean, [2.5], what="the mean")
    close(scale, [np.sqrt(1.25)], what="the scale")


@check("edge", "columns with very different sizes", "Standardizing makes columns that differ by a factor of one million comparable.")
def different_sizes(sub):
    rng = np.random.default_rng(4)
    x = np.column_stack([rng.normal(size=30), rng.normal(size=30) * 1e6 + 1e9])
    mean, scale = sub.fit_standardizer(x)
    z = sub.apply_standardizer(x, mean, scale)
    close(np.std(z, axis=0), np.ones(2), atol=1e-6, what="the column standard deviations")


@fixture("data")
def data(ref):
    rng = np.random.default_rng(42)
    for n, d in [(2, 1), (5, 3), (30, 4), (7, 6)]:
        x = rng.normal(size=(n, d)) * rng.uniform(0.1, 50, size=d) + rng.uniform(-100, 100, size=d)
        x[:, 0] = x[0, 0]
        new = rng.normal(size=(3, d))
        mean, scale = ref.fit_standardizer(x)
        yield {"x": x, "new": new}, {"mean": mean, "scale": scale, "train": ref.apply_standardizer(x, mean, scale), "new": ref.apply_standardizer(new, mean, scale)}


@check("reference", "fit and apply match the reference", "Your statistics and standardized data agree with the reference on random data that includes a constant column.")
def reference_all(sub):
    for case, expected in fixtures("data"):
        mean, scale = sub.fit_standardizer(case["x"])
        close(mean, expected["mean"], rtol=1e-9, atol=1e-12, what="the mean")
        close(scale, expected["scale"], rtol=1e-9, atol=1e-12, what="the scale")
        close(sub.apply_standardizer(case["x"], mean, scale), expected["train"], rtol=1e-9, atol=1e-9, what="the standardized train data")
        close(sub.apply_standardizer(case["new"], mean, scale), expected["new"], rtol=1e-9, atol=1e-9, what="the standardized new data")
