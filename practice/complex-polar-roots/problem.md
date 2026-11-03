A complex number $x + iy$ is also a point in the plane, with a radius $r$ and an angle $\theta$. In this form, multiplication is simple: the radii multiply and the angles add. This makes powers and roots easy, and it is the reason why complex numbers appear in rotations, in signals and in the Fourier transform.

## Task

Write five functions. Work with the real and imaginary parts, and do not use `np.abs`, `np.angle` or `**` on complex numbers.

1. `to_polar(z)` returns `(r, theta)`. The radius is $r = \sqrt{x^2 + y^2}$ (`np.hypot`). The angle is $\theta = \operatorname{atan2}(y, x)$ (`np.arctan2`).
2. `from_polar(r, theta)` returns the complex number $r\cos\theta + i\, r\sin\theta$.
3. `complex_power(z, n)` returns $z^n$ for a whole number $n \ge 0$ with de Moivre's formula: radius $r^n$ and angle $n\theta$.
4. `roots_of_unity(n)` returns the $n$ complex numbers $e^{2\pi i k / n}$ for $k = 0, \dots, n-1$.
5. `nth_roots(z, n)` returns all $n$ roots of $z$: radius $r^{1/n}$ and angles $(\theta + 2\pi k)/n$ for $k = 0, \dots, n-1$.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `z` | complex, real, or complex array | scalar or any shape | the number or numbers |
| `r`, `theta` | float or float array | same shape | the radius and the angle in radians |
| `n` | int | scalar | the power, or the number of roots. `n >= 0` for the power and `n >= 1` for roots |

- `to_polar` returns two float arrays. The other functions return complex arrays. `roots_of_unity` and `nth_roots` return the shape `(n,)`.

## Constraints

- Do not change the inputs.
- Use numpy only.

## Examples

```python
r, theta = to_polar(1 + 1j)
float(r)
float(theta)
```

```text
1.4142135623730951
0.7853981633974483
```

```python
complex(np.round(from_polar(2.0, np.pi / 2), 6))
round(float(complex_power(1 + 1j, 8).real), 6)
```

```text
2j
16.0
```

```python
np.round(np.angle(roots_of_unity(4)), 6)
bool(np.allclose(nth_roots(-8, 3) ** 3, -8))
```

```text
array([ 0.      ,  1.570796,  3.141593, -1.570796])
True
```
