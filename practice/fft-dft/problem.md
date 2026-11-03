The discrete Fourier transform (DFT) writes a signal as a sum of waves. Entry $X_k$ tells you how much of the wave with $k$ periods the signal contains. The direct sum costs $n^2$ operations. The fast Fourier transform (FFT) computes the same numbers in $n \log n$ operations, by splitting the signal into its even and odd samples.

## Task

Write three functions. The signal can be real or complex.

1. `dft(x)` returns the spectrum by the direct sum:

$$X_k = \sum_{j=0}^{n-1} x_j\, e^{-2\pi i\, jk / n}, \qquad k = 0, \dots, n-1$$

2. `fft(x)` returns the same spectrum with the recursive radix-2 algorithm. Raise `ValueError` if the length is not a power of 2 (the lengths 1, 2, 4, 8, ...). The recursion:
   1. If $n = 1$, return the signal.
   2. Let $E$ be the FFT of the even-indexed samples and $O$ the FFT of the odd-indexed samples. Each has length $n/2$.
   3. Let $t_k = e^{-2\pi i\, k/n}\, O_k$ for $k = 0, \dots, n/2 - 1$.
   4. Return $X_k = E_k + t_k$ for the first half and $X_{k + n/2} = E_k - t_k$ for the second half.
3. `ifft(spectrum)` returns the signal for a spectrum: the conjugate of `fft` of the conjugated spectrum, divided by $n$.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `x` | float or complex array | `(n,)` | the signal. For `fft`, `n` is a power of 2 |
| `spectrum` | complex array | `(n,)` | the spectrum. For `ifft`, `n` is a power of 2 |

- All functions return a complex array with the shape `(n,)`.

## Constraints

- Do not use `np.fft` or other library transforms.
- Do not change the input.
- Use numpy only.

## Examples

```python
x = np.array([1.0, 2.0, 3.0, 4.0])
np.round(dft(x).real, 6) + 0.0
np.round(dft(x).imag, 6) + 0.0
bool(np.allclose(fft(x), dft(x)))
bool(np.allclose(ifft(fft(x)), x))
```

```text
array([10., -2., -2., -2.])
array([ 0.,  2.,  0., -2.])
True
True
```
