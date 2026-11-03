from functools import reduce

import numpy as np

from practice_check import check, close, equal, fixture, fixtures, raises, true

START = (0, 0.0, 0.0)
DATA = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]


def fold(sub, values, state=START):
    return reduce(sub.welford_update, values, state)


@check("examples", "fold a data set into a state", "The state holds the count, the mean and the sum of squared distances to the mean (M2). For this data they are 8, 5 and 32.")
def example_update(sub):
    n, mean, m2 = fold(sub, DATA)
    equal(n, 8, what="the count")
    close(mean, 5.0, what="the mean")
    close(m2, 32.0, what="M2")


@check("examples", "mean and variance from the state", "The population variance is M2 / n = 4. With ddof 1 it is M2 / (n - 1).")
def example_finalize(sub):
    state = fold(sub, DATA)
    mean, var = sub.welford_finalize(state)
    close((mean, var), (5.0, 4.0), what="(mean, variance)")
    close(sub.welford_finalize(state, ddof=1)[1], 32.0 / 7.0, rtol=1e-12, what="the sample variance")


@check("examples", "merge two halves", "Merging the states of the two halves gives the state of the whole data set.")
def example_merge(sub):
    merged = sub.welford_merge(fold(sub, DATA[:4]), fold(sub, DATA[4:]))
    close(merged, fold(sub, DATA), rtol=1e-12, what="the merged state")


@check("properties", "the result matches numpy", "The mean and the variance agree with np.mean and np.var for ddof 0 and 1.")
def matches_numpy(sub):
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=500)
    state = fold(sub, list(x))
    for ddof in (0, 1):
        mean, var = sub.welford_finalize(state, ddof=ddof)
        close(mean, np.mean(x), rtol=1e-10, what="the mean")
        close(var, np.var(x, ddof=ddof), rtol=1e-10, what=f"the variance with ddof={ddof}")


@check("properties", "merging any split gives the state of the whole", "The count, the mean and M2 of a merged pair equal the state of all the data, for any split point.")
def merge_splits(sub):
    rng = np.random.default_rng(1)
    x = list(rng.normal(size=60))
    whole = fold(sub, x)
    for cut in (1, 10, 30, 59):
        close(sub.welford_merge(fold(sub, x[:cut]), fold(sub, x[cut:])), whole, rtol=1e-9, atol=1e-12, what=f"the merged state for a cut at {cut}")


@check("properties", "merge is commutative and associative", "The order of the parts does not matter, so parts can be computed on different machines.")
def merge_laws(sub):
    rng = np.random.default_rng(2)
    a, b, c = (fold(sub, list(rng.normal(loc=i, size=20 + i))) for i in range(3))
    close(sub.welford_merge(a, b), sub.welford_merge(b, a), rtol=1e-9, atol=1e-12, what="merge(a, b) against merge(b, a)")
    close(sub.welford_merge(sub.welford_merge(a, b), c), sub.welford_merge(a, sub.welford_merge(b, c)), rtol=1e-9, atol=1e-12, what="the two groupings")


@check("properties", "one update equals a merge with a single-value state", "A state for one value x is (1, x, 0). Merging it adds that value.")
def update_is_merge(sub):
    state = fold(sub, [1.0, 5.0, 2.0])
    close(sub.welford_update(state, 4.0), sub.welford_merge(state, (1, 4.0, 0.0)), rtol=1e-12, what="the new state")


@check("edge", "values with a huge offset keep their variance", "The values are 1e9 plus a small spread. The formula mean of squares minus square of mean loses all digits here. The Welford update does not.")
def large_offset(sub):
    rng = np.random.default_rng(3)
    noise = rng.normal(size=1000)
    x = 1e9 + noise
    mean, var = sub.welford_finalize(fold(sub, list(x)))
    close(var, np.var(noise), rtol=1e-6, what="the variance")


@check("edge", "one value has variance 0", "A single value has no spread.")
def single_value(sub):
    mean, var = sub.welford_finalize(fold(sub, [3.5]))
    close((mean, var), (3.5, 0.0), what="(mean, variance)")


@check("edge", "too few values raise ValueError", "The sample variance needs at least 2 values. An empty state has no mean.")
def too_few(sub):
    raises(ValueError, sub.welford_finalize, fold(sub, [3.5]), 1)
    raises(ValueError, sub.welford_finalize, START)


@check("edge", "merging with an empty state", "An empty state changes nothing. Two empty states give an empty state without a division by zero.")
def merge_empty(sub):
    state = fold(sub, [1.0, 2.0, 6.0])
    close(sub.welford_merge(state, START), state, rtol=1e-12, what="state merged with empty")
    close(sub.welford_merge(START, state), state, rtol=1e-12, what="empty merged with state")
    close(sub.welford_merge(START, START), START, atol=0, what="empty merged with empty")


@fixture("streams")
def streams(ref):
    rng = np.random.default_rng(42)
    for n, loc, scale in [(5, 0.0, 1.0), (50, 100.0, 0.1), (300, -5.0, 20.0)]:
        x = rng.normal(loc=loc, scale=scale, size=n)
        state = reduce(ref.welford_update, list(x), (0, 0.0, 0.0))
        cut = n // 3
        merged = ref.welford_merge(reduce(ref.welford_update, list(x[:cut]), (0, 0.0, 0.0)), reduce(ref.welford_update, list(x[cut:]), (0, 0.0, 0.0)))
        yield {"x": x, "cut": cut}, {"state": list(state), "merged": list(merged), "final": list(ref.welford_finalize(state, ddof=1))}


@check("reference", "states and results match the reference", "Your update, merge and finalize agree with the reference on three streams.")
def reference_all(sub):
    for case, expected in fixtures("streams"):
        x = list(case["x"])
        state = fold(sub, x)
        close(state, expected["state"], rtol=1e-9, atol=1e-12, what="the state")
        merged = sub.welford_merge(fold(sub, x[:case["cut"]]), fold(sub, x[case["cut"]:]))
        close(merged, expected["merged"], rtol=1e-9, atol=1e-12, what="the merged state")
        close(sub.welford_finalize(state, ddof=1), expected["final"], rtol=1e-9, atol=1e-12, what="mean and sample variance")
