import numpy as np

from practice_check import check, close, equal, fixture, fixtures, shape

X = np.arange(16, dtype=float).reshape(1, 4, 4)
W = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])


@check("examples", "output size formula", "The output size is floor((size + 2 * padding - kernel) / stride) + 1.")
def example_output_size(sub):
    equal(sub.output_size(32, 3, 1, 1), 32, what="output_size(32, 3, 1, 1)")
    equal(sub.output_size(7, 3, 0, 2), 3, what="output_size(7, 3, 0, 2)")


@check("examples", "stride 1, no padding", "A 2x2 kernel on a 4x4 image gives a 3x3 output. This kernel computes top-left minus bottom-right, which is -5 everywhere.")
def example_plain(sub):
    close(sub.conv2d(X, W), np.full((1, 3, 3), -5.0), what="the output")


@check("examples", "stride 2", "With stride 2 the window moves two pixels at a time, so the output is 2x2.")
def example_stride(sub):
    close(sub.conv2d(X, W, stride=2), np.full((1, 2, 2), -5.0), what="the output")


@check("examples", "padding 1 and a bias", "Padding 1 adds a ring of zeros, so the output is 5x5. The bias 0.5 is added to every position.")
def example_pad_bias(sub):
    expected = [[[0.5, -0.5, -1.5, -2.5, 0.5],
                 [-3.5, -4.5, -4.5, -4.5, 3.5],
                 [-7.5, -4.5, -4.5, -4.5, 7.5],
                 [-11.5, -4.5, -4.5, -4.5, 11.5],
                 [0.5, 12.5, 13.5, 14.5, 15.5]]]
    close(sub.conv2d(X, W, b=np.array([0.5]), padding=1), expected, what="the output")


@check("properties", "output shape is (C_out, H_out, W_out)", "The shape follows from the kernel count and from output_size on each spatial axis.")
def output_shape(sub):
    rng = np.random.default_rng(0)
    for c_in, c_out, h, w_, k, s, p in [(3, 4, 9, 7, 3, 1, 1), (2, 5, 10, 10, 3, 2, 0), (1, 1, 8, 6, 5, 2, 2)]:
        x = rng.normal(size=(c_in, h, w_))
        kernels = rng.normal(size=(c_out, c_in, k, k))
        out = sub.conv2d(x, kernels, stride=s, padding=p)
        shape(out, (c_out, (h + 2 * p - k) // s + 1, (w_ + 2 * p - k) // s + 1), what="the output")


@check("properties", "a 1x1 identity kernel returns the image", "A 1x1 kernel with weight 1 from the same channel copies the image.")
def identity_kernel(sub):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 5, 6))
    kernels = np.zeros((3, 3, 1, 1))
    for c in range(3):
        kernels[c, c, 0, 0] = 1.0
    close(sub.conv2d(x, kernels), x, atol=1e-12, what="the output")


@check("properties", "the layer is linear in the input", "conv2d(a*x + y) equals a*conv2d(x) + conv2d(y) when there is no bias.")
def linear_in_input(sub):
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(2, 3, 6, 6))
    kernels = rng.normal(size=(2, 3, 3, 3))
    left = sub.conv2d(2.5 * x + y, kernels, padding=1)
    right = 2.5 * sub.conv2d(x, kernels, padding=1) + sub.conv2d(y, kernels, padding=1)
    close(left, right, atol=1e-9, what="the two sides")


@check("properties", "a bias adds one constant to each output channel", "The bias b[o] is added to every position of output channel o.")
def bias_constant(sub):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 5, 5))
    kernels = rng.normal(size=(3, 2, 3, 3))
    bias = np.array([1.0, -2.0, 0.25])
    diff = sub.conv2d(x, kernels, b=bias, padding=1) - sub.conv2d(x, kernels, padding=1)
    close(diff, np.broadcast_to(bias[:, None, None], diff.shape), atol=1e-9, what="output with bias minus output without bias")


@check("properties", "shifting the input shifts the output", "Convolution is translation equivariant. Away from the border, moving the image by one pixel moves the output by one pixel.")
def shift_equivariance(sub):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(1, 10, 10))
    kernels = rng.normal(size=(1, 1, 3, 3))
    shifted = np.zeros_like(x)
    shifted[:, 1:, :] = x[:, :-1, :]
    base = sub.conv2d(x, kernels)
    moved = sub.conv2d(shifted, kernels)
    close(moved[:, 1:, :], base[:, :-1, :], atol=1e-9, what="the shifted output")


@check("properties", "the inputs are not changed", "A function that writes into its input arrays breaks the code that calls it.")
def inputs_unchanged(sub):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 5, 5))
    kernels = rng.normal(size=(2, 2, 3, 3))
    bias = rng.normal(size=2)
    saved = (x.copy(), kernels.copy(), bias.copy())
    sub.conv2d(x, kernels, b=bias, stride=2, padding=1)
    close(x, saved[0], atol=0, rtol=0, what="x")
    close(kernels, saved[1], atol=0, rtol=0, what="w")
    close(bias, saved[2], atol=0, rtol=0, what="b")


@check("properties", "the kernel is not flipped", "A kernel with one 1 at the top-left corner must copy the top-left pixel of each window.")
def not_flipped(sub):
    kernels = np.zeros((1, 1, 2, 2))
    kernels[0, 0, 0, 0] = 1.0
    close(sub.conv2d(X, kernels), X[:, :3, :3], what="the output")


@check("edge", "kernel as large as the image", "When the kernel is as large as the image, there is exactly one output position and it is a dot product.")
def full_size_kernel(sub):
    rng = np.random.default_rng(6)
    x = rng.normal(size=(2, 4, 4))
    kernels = rng.normal(size=(3, 2, 4, 4))
    expected = np.einsum("ocuv,cuv->o", kernels, x).reshape(3, 1, 1)
    close(sub.conv2d(x, kernels), expected, atol=1e-9, what="the output")


@check("edge", "stride larger than the kernel skips pixels", "With stride 3 and kernel 2, one column between the windows is never read.")
def stride_skips(sub):
    x = np.arange(25, dtype=float).reshape(1, 5, 5)
    kernels = np.ones((1, 1, 2, 2))
    expected = [[[x[0, 0:2, 0:2].sum(), x[0, 0:2, 3:5].sum()], [x[0, 3:5, 0:2].sum(), x[0, 3:5, 3:5].sum()]]]
    close(sub.conv2d(x, kernels, stride=3), expected, what="the output")


@check("edge", "a window that covers only padding returns the bias", "A window over zero padding sees only zeros, so the output there is the bias.")
def padding_only_window(sub):
    x = np.ones((1, 2, 2))
    kernels = np.ones((1, 1, 2, 2))
    out = sub.conv2d(x, kernels, b=np.array([3.0]), padding=2)
    shape(out, (1, 5, 5), what="the output")
    close(out[0, 0, 0], 3.0, what="the top-left value")
    close(out[0, 2, 2], 7.0, what="the value where the window covers the whole image")


@fixture("cases")
def cases(ref):
    rng = np.random.default_rng(42)
    for c_in, c_out, h, w_, k, s, p, use_b in [(1, 1, 5, 5, 3, 1, 0, False), (3, 2, 8, 6, 3, 1, 1, True), (2, 4, 9, 9, 3, 2, 1, True), (3, 3, 7, 10, 5, 2, 2, False), (1, 2, 6, 6, 1, 1, 0, True)]:
        x = rng.normal(size=(c_in, h, w_))
        kernels = rng.normal(size=(c_out, c_in, k, k))
        bias = rng.normal(size=c_out) if use_b else None
        yield {"x": x, "w": kernels, "b": bias, "stride": s, "padding": p}, ref.conv2d(x, kernels, bias, s, p)


@check("reference", "conv2d matches the reference on random layers", "Your layer gives the same values as the reference on several shapes, strides and paddings.")
def reference_conv(sub):
    for case, expected in fixtures("cases"):
        out = sub.conv2d(case["x"], case["w"], case["b"], case["stride"], case["padding"])
        close(out, expected, rtol=1e-7, atol=1e-9, what=f"conv2d for x {case['x'].shape}, stride {case['stride']}, padding {case['padding']}")
