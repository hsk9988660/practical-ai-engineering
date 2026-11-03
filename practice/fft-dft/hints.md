## Hint 1

For `dft`, build the matrix with entries $e^{-2\pi i\, jk/n}$ using `np.outer(k, k)`. The spectrum is the matrix times the signal.

## Hint 2

For `fft`, check the length first. The test `n & (n - 1) == 0` is true only for powers of 2, and you must also reject $n = 0$.

## Hint 3

The recursion uses `x[0::2]` for the even samples and `x[1::2]` for the odd samples. Each half has length `n // 2`.

## Hint 4

The twiddle factors are `np.exp(-2j * np.pi * np.arange(n // 2) / n)`. Multiply them by the FFT of the odd part. Then join `even + twiddle` and `even - twiddle` with `np.concatenate`.

## Hint 5

The inverse transform is the same algorithm with the opposite sign in the exponent. Conjugating the input and the output gives this sign flip without new code.
