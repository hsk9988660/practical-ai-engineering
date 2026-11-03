## What the numbers mean

Entry $X_k$ is the correlation of the signal with a wave that makes $k$ turns over the signal. A large $|X_k|$ means that the signal contains a lot of that frequency. $X_0$ is the sum of the signal. For a real signal, the second half of the spectrum is the mirror of the first half.

## Why the FFT is fast

The DFT of a signal of length $n$ can be written with two DFTs of length $n/2$: one for the even samples and one for the odd samples. The two results are combined with $n$ multiplications by the twiddle factors. Repeating this $\log_2 n$ times gives $n \log_2 n$ operations. For $n = 10^6$ this is 50000 times fewer than $n^2$.

## Sign and scale conventions

Libraries differ: some put the minus sign in the forward transform, some in the inverse, and some put the factor $1/n$ in the forward transform or use $1/\sqrt{n}$ on both sides. This exercise follows numpy: a minus sign forward and a factor $1/n$ in the inverse.

## Other lengths

The radix-2 algorithm needs a power of 2. Real libraries also handle other lengths, with other factors or with a padding trick. For a signal of another length you can pad with zeros, which changes the frequency grid, or use the direct sum.

## Where this goes next

Convolution in time is multiplication in frequency, which is how large convolutions in signal processing and in some neural networks are computed fast. Audio models use the short-time Fourier transform to make spectrograms, in the speech phase.
