import numpy as np

from practice_check import check, close, fixture, fixtures, raises, shape, true


def cclose(actual, expected, rtol=1e-6, atol=1e-8, what="the output"):
    close(np.real(actual), np.real(expected), rtol=rtol, atol=atol, what=f"the real part of {what}")
    close(np.imag(actual), np.imag(expected), rtol=rtol, atol=atol, what=f"the imaginary part of {what}")


X4 = np.array([1.0, 2.0, 3.0, 4.0])


@check("examples", "DFT of [1, 2, 3, 4]", "The first value is the sum, 10. The others are -2 + 2i, -2 and -2 - 2i.")
def example_dft(sub):
    cclose(sub.dft(X4), [10, -2 + 2j, -2, -2 - 2j], atol=1e-9, what="the spectrum")


@check("examples", "FFT gives the same spectrum", "The fast algorithm computes the same numbers as the direct sum, with fewer operations.")
def example_fft(sub):
    cclose(sub.fft(X4), [10, -2 + 2j, -2, -2 - 2j], atol=1e-9, what="the spectrum")


@check("examples", "the inverse transform returns the signal", "ifft(fft(x)) is x. The result is complex, with imaginary part 0 for a real signal.")
def example_ifft(sub):
    cclose(sub.ifft(sub.fft(X4)), X4, atol=1e-9, what="the signal")


@check("properties", "dft and fft match numpy", "Your transforms agree with np.fft.fft on random signals of several lengths.")
def match_numpy(sub):
    rng = np.random.default_rng(0)
    for n in (1, 2, 4, 8, 32, 64):
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        cclose(sub.fft(x), np.fft.fft(x), rtol=1e-9, atol=1e-9, what=f"fft for n={n}")
        cclose(sub.dft(x), np.fft.fft(x), rtol=1e-9, atol=1e-9, what=f"dft for n={n}")


@check("properties", "dft works for lengths that are not powers of 2", "The direct sum has no restriction on the length.")
def dft_any_length(sub):
    rng = np.random.default_rng(1)
    for n in (3, 5, 6, 7, 10):
        x = rng.normal(size=n)
        cclose(sub.dft(x), np.fft.fft(x), rtol=1e-9, atol=1e-9, what=f"dft for n={n}")


@check("properties", "the round trip ifft(fft(x)) returns x", "The inverse transform undoes the transform, including the factor 1 / n.")
def round_trip(sub):
    rng = np.random.default_rng(2)
    x = rng.normal(size=16) + 1j * rng.normal(size=16)
    cclose(sub.ifft(sub.fft(x)), x, atol=1e-9, what="the round trip")


@check("properties", "Parseval: the energy is the same in both domains", "The sum of |x|^2 equals the sum of |X|^2 divided by n.")
def parseval(sub):
    rng = np.random.default_rng(3)
    x = rng.normal(size=32)
    spectrum = sub.fft(x)
    close(np.sum(np.abs(x) ** 2), np.sum(np.abs(spectrum) ** 2) / 32, rtol=1e-9, what="the energy")


@check("properties", "the transform is linear", "fft(a x + y) equals a fft(x) + fft(y).")
def linear(sub):
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=16), rng.normal(size=16)
    cclose(sub.fft(2.5 * x + y), 2.5 * sub.fft(x) + sub.fft(y), rtol=1e-9, atol=1e-9, what="the two sides")


@check("properties", "a real signal has a conjugate-symmetric spectrum", "For a real x, X[n - k] is the complex conjugate of X[k].")
def conjugate_symmetry(sub):
    rng = np.random.default_rng(5)
    spectrum = sub.fft(rng.normal(size=16))
    cclose(spectrum[1:][::-1], np.conj(spectrum[1:]), atol=1e-9, what="the upper half of the spectrum")


@check("properties", "a shift in time is a phase in frequency", "Rolling the signal by s multiplies X[k] by exp(-2 pi i k s / n).")
def shift_theorem(sub):
    rng = np.random.default_rng(6)
    n, s = 16, 3
    x = rng.normal(size=n)
    phase = np.exp(-2j * np.pi * np.arange(n) * s / n)
    cclose(sub.fft(np.roll(x, s)), sub.fft(x) * phase, atol=1e-9, what="the spectrum of the shifted signal")


@check("properties", "a sine wave has its energy in one frequency", "A sine with 3 full periods in 32 samples gives peaks at bins 3 and 29 only.")
def sine_peak(sub):
    n = 32
    x = np.sin(2 * np.pi * 3 * np.arange(n) / n)
    magnitude = np.abs(sub.fft(x))
    close(magnitude[3], n / 2, rtol=1e-9, what="the peak at bin 3")
    close(magnitude[29], n / 2, rtol=1e-9, what="the peak at bin 29")
    other = np.delete(magnitude, [3, 29])
    close(other, np.zeros(30), atol=1e-9, what="the other bins")


@check("properties", "the input is not changed", "A function that writes into its input arrays breaks the code that calls it.")
def input_unchanged(sub):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    saved = x.copy()
    sub.fft(x)
    sub.dft(x)
    sub.ifft(x)
    close(x, saved, atol=0, rtol=0, what="x")


@check("edge", "an impulse and a constant", "An impulse at time 0 has a flat spectrum of ones. A constant signal has all its energy in bin 0.")
def impulse_constant(sub):
    impulse = np.zeros(8)
    impulse[0] = 1.0
    cclose(sub.fft(impulse), np.ones(8), atol=1e-12, what="the spectrum of the impulse")
    expected = np.zeros(8)
    expected[0] = 8.0
    cclose(sub.fft(np.ones(8)), expected, atol=1e-12, what="the spectrum of the constant")


@check("edge", "one sample", "The transform of a single sample is the sample itself.")
def one_sample(sub):
    cclose(sub.fft(np.array([3.5])), [3.5], atol=1e-12, what="the spectrum")


@check("edge", "fft raises ValueError for a length that is not a power of 2", "The radix-2 algorithm splits the signal in halves, so the length must be a power of 2. Use dft for other lengths.")
def not_power_of_two(sub):
    raises(ValueError, sub.fft, np.ones(6))
    raises(ValueError, sub.fft, np.ones(12))
    raises(ValueError, sub.fft, np.ones(0))


@fixture("signals")
def signals(ref):
    rng = np.random.default_rng(42)
    for n in (2, 8, 16, 64):
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        f, d, i = ref.fft(x), ref.dft(x), ref.ifft(x)
        yield {"re": x.real, "im": x.imag}, {"f_re": f.real, "f_im": f.imag, "d_re": d.real, "d_im": d.imag, "i_re": i.real, "i_im": i.imag}


@check("reference", "fft, dft and ifft match the reference", "Your three transforms equal the reference on random complex signals.")
def reference_all(sub):
    for case, expected in fixtures("signals"):
        x = np.array(case["re"]) + 1j * np.array(case["im"])
        cclose(sub.fft(x), np.array(expected["f_re"]) + 1j * np.array(expected["f_im"]), rtol=1e-9, atol=1e-9, what=f"fft for n={len(x)}")
        cclose(sub.dft(x), np.array(expected["d_re"]) + 1j * np.array(expected["d_im"]), rtol=1e-9, atol=1e-9, what=f"dft for n={len(x)}")
        cclose(sub.ifft(x), np.array(expected["i_re"]) + 1j * np.array(expected["i_im"]), rtol=1e-9, atol=1e-9, what=f"ifft for n={len(x)}")
