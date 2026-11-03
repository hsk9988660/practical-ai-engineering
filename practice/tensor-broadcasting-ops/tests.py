import numpy as np

from practice_check import check, close, fixture, fixtures, raises, shape, true


@check("examples", "normalize each channel", "The mean and std have one number for each channel. Each channel is shifted and scaled by its own numbers.")
def example_normalize(sub):
    x = np.array([[[[1.0, 3.0]], [[10.0, 30.0]]]])
    out = sub.channel_normalize(x, np.array([2.0, 20.0]), np.array([1.0, 10.0]))
    close(out, [[[[-1.0, 1.0]], [[-1.0, 1.0]]]], atol=1e-12, what="the normalized images")


@check("examples", "one-hot encode labels", "Row i has a 1 at the column of label i and 0 elsewhere.")
def example_one_hot(sub):
    close(sub.one_hot(np.array([2, 0]), 3), [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], atol=0, what="the encoding")


@check("examples", "a batch of outer products", "For each example, entry (i, j) is a[i] times b[j].")
def example_outer(sub):
    close(sub.batched_outer(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0, 5.0]])), [[[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]], atol=1e-12, what="the outer products")


@check("examples", "a masked mean over time", "Only the first two time steps are valid, so the mean is taken over those two. The step with 100 and 200 is ignored.")
def example_masked(sub):
    x = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
    close(sub.masked_mean(x, np.array([[True, True, False]])), [[2.0, 3.0]], atol=1e-12, what="the means")


@check("properties", "channel_normalize works for any channel count and image size", "The channel axis is axis 1. A code that broadcasts on the wrong axis fails when C is not equal to W.")
def normalize_shapes(sub):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 5, 4))
    mean = np.array([0.1, 0.2, 0.3])
    std = np.array([1.0, 2.0, 4.0])
    out = sub.channel_normalize(x, mean, std)
    shape(out, (2, 3, 5, 4), what="the output")
    for c in range(3):
        close(out[:, c], (x[:, c] - mean[c]) / std[c], atol=1e-12, what=f"channel {c}")


@check("properties", "normalized channels have mean 0 and std 1 when the true statistics are used", "If you pass the real per-channel mean and std, the result is standardized.")
def normalize_standardizes(sub):
    rng = np.random.default_rng(1)
    x = rng.normal(loc=[[[[5.0]], [[-3.0]], [[0.0]]]], scale=[[[[1.0]], [[4.0]], [[0.5]]]], size=(8, 3, 6, 6))
    mean = x.mean(axis=(0, 2, 3))
    std = x.std(axis=(0, 2, 3))
    out = sub.channel_normalize(x, mean, std)
    close(out.mean(axis=(0, 2, 3)), np.zeros(3), atol=1e-9, what="the channel means")
    close(out.std(axis=(0, 2, 3)), np.ones(3), atol=1e-9, what="the channel stds")


@check("properties", "one_hot rows sum to 1 and have the right shape", "Each row has exactly one 1.")
def one_hot_rows(sub):
    labels = np.array([0, 4, 2, 2, 1])
    out = sub.one_hot(labels, 5)
    shape(out, (5, 5), what="the encoding")
    close(out.sum(axis=1), np.ones(5), what="the row sums")
    close(np.argmax(out, axis=1), labels, what="the position of the 1")


@check("properties", "batched_outer matches a loop", "Example b of the output is the outer product of a[b] and b[b].")
def outer_loop(sub):
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(4, 3)), rng.normal(size=(4, 5))
    out = sub.batched_outer(a, b)
    shape(out, (4, 3, 5), what="the output")
    close(out, np.array([np.outer(a[i], b[i]) for i in range(4)]), atol=1e-12, what="the outer products")


@check("properties", "masked_mean equals a loop over the valid steps", "Each example has its own number of valid steps.")
def masked_loop(sub):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 6, 3))
    mask = rng.random((5, 6)) > 0.4
    mask[:, 0] = True
    expected = np.array([x[i][mask[i]].mean(axis=0) for i in range(5)])
    close(sub.masked_mean(x, mask), expected, atol=1e-12, what="the means")


@check("properties", "the inputs are not changed", "A function that writes into its input arrays breaks the code that calls it.")
def inputs_unchanged(sub):
    x = np.ones((2, 3, 2, 2))
    mean, std = np.zeros(3), np.ones(3)
    seq = np.ones((2, 4, 3))
    mask = np.array([[True, False, True, True], [False, False, True, True]])
    saved = (x.copy(), seq.copy(), mask.copy())
    sub.channel_normalize(x, mean, std)
    sub.masked_mean(seq, mask)
    close(x, saved[0], atol=0, rtol=0, what="x")
    close(seq, saved[1], atol=0, rtol=0, what="the sequence")
    true(bool(np.array_equal(mask, saved[2])), "the mask changed", "the same mask", mask)


@check("edge", "an example with no valid step gets zeros", "If the mask is all False, there is nothing to average. The result is 0, not NaN.")
def no_valid_step(sub):
    x = np.ones((2, 3, 2))
    mask = np.array([[True, True, False], [False, False, False]])
    out = sub.masked_mean(x, mask)
    close(out, [[1.0, 1.0], [0.0, 0.0]], atol=0, what="the means")


@check("edge", "a label outside the range raises ValueError", "A negative label would wrap around to the last column, and a label that is too large is an error.")
def label_range(sub):
    raises(ValueError, sub.one_hot, np.array([0, 3]), 3)
    raises(ValueError, sub.one_hot, np.array([-1, 1]), 3)


@check("edge", "empty label list", "No labels give an array of shape (0, n).")
def empty_labels(sub):
    shape(sub.one_hot(np.array([], dtype=int), 4), (0, 4), what="the encoding")


@fixture("cases")
def cases(ref):
    rng = np.random.default_rng(42)
    for n, c, h, w in [(1, 1, 2, 2), (2, 3, 4, 5), (3, 2, 6, 3)]:
        x = rng.normal(size=(n, c, h, w))
        mean, std = rng.normal(size=c), rng.uniform(0.5, 2, size=c)
        seq = rng.normal(size=(n, h, c))
        mask = rng.random((n, h)) > 0.3
        labels = rng.integers(0, 6, size=n)
        a, b = rng.normal(size=(n, c)), rng.normal(size=(n, w))
        yield {"x": x, "mean": mean, "std": std, "seq": seq, "mask": mask, "labels": labels, "a": a, "b": b}, {
            "norm": ref.channel_normalize(x, mean, std), "mm": ref.masked_mean(seq, mask), "oh": ref.one_hot(labels, 6), "outer": ref.batched_outer(a, b)}


@check("reference", "all four functions match the reference", "Your outputs equal the reference on random tensors of several shapes.")
def reference_all(sub):
    for case, expected in fixtures("cases"):
        close(sub.channel_normalize(np.array(case["x"]), np.array(case["mean"]), np.array(case["std"])), expected["norm"], rtol=1e-9, atol=1e-12, what="channel_normalize")
        close(sub.masked_mean(np.array(case["seq"]), np.array(case["mask"])), expected["mm"], rtol=1e-9, atol=1e-12, what="masked_mean")
        close(sub.one_hot(np.array(case["labels"]), 6), expected["oh"], atol=0, what="one_hot")
        close(sub.batched_outer(np.array(case["a"]), np.array(case["b"])), expected["outer"], rtol=1e-9, atol=1e-12, what="batched_outer")
