## Why polar form

Adding is easy in the form $x + iy$. Multiplying is easy in polar form: $(r_1, \theta_1)(r_2, \theta_2) = (r_1 r_2, \theta_1 + \theta_2)$. A multiplication by a complex number is a rotation and a scaling of the plane. This is the 2D rotation matrix in another form.

## atan2

The angle of a point depends on the signs of both coordinates. $\arctan(y/x)$ only returns angles between $-90$ and $90$ degrees, so it cannot tell the point $(1, 1)$ from $(-1, -1)$. `arctan2` takes both numbers and returns an angle between $-\pi$ and $\pi$.

## Roots

A non-zero number has exactly $n$ different $n$-th roots. They sit on a circle and are spaced by $2\pi/n$. The roots of 1 are the roots of unity. They add up to 0, and they are the building block of the discrete Fourier transform.

## Floating point

Computing $\cos(2\pi)$ gives $1$ and $\sin(2\pi)$ gives $-2.4 \cdot 10^{-16}$, not 0. Compare complex results with a tolerance.

## Where this goes next

The next exercise builds the discrete Fourier transform from the roots of unity. Rotary position embeddings in transformers rotate pairs of numbers by an angle, which is a multiplication by a complex number of radius 1.
