import numpy as np

from practice_check import check, close, fixture, fixtures, shape, true


def cclose(actual, expected, rtol=1e-6, atol=1e-8, what="the output"):
    close(np.real(actual), np.real(expected), rtol=rtol, atol=atol, what=f"the real part of {what}")
    close(np.imag(actual), np.imag(expected), rtol=rtol, atol=atol, what=f"the imaginary part of {what}")


@check("examples", "polar form of 1 + i", "The radius is sqrt(2) and the angle is 45 degrees, which is pi / 4.")
def example_polar(sub):
    r, theta = sub.to_polar(1 + 1j)
    cclose(r, np.sqrt(2.0), rtol=1e-12, what="the radius")
    cclose(theta, np.pi / 4, rtol=1e-12, what="the angle")


@check("examples", "from polar form", "2 times (cos 90 degrees + i sin 90 degrees) is 2i.")
def example_from_polar(sub):
    cclose(sub.from_polar(2.0, np.pi / 2), 2j, atol=1e-12, what="the number")


@check("examples", "a power by de Moivre", "(1 + i)^8 has radius sqrt(2)^8 = 16 and angle 8 * 45 degrees = 360 degrees, so it is 16.")
def example_power(sub):
    cclose(sub.complex_power(1 + 1j, 8), 16 + 0j, atol=1e-9, what="the power")


@check("examples", "the fourth roots of unity", "The 4 roots lie on the unit circle at the angles 0, 90, 180 and 270 degrees.")
def example_roots(sub):
    roots = sub.roots_of_unity(4)
    shape(roots, (4,), what="the roots")
    cclose(roots, [1, 1j, -1, -1j], atol=1e-12, what="the roots")


@check("examples", "the cube roots of -8", "The real root is -2. The other two are complex and have the same radius 2.")
def example_nth_roots(sub):
    roots = sub.nth_roots(-8, 3)
    cclose(np.sort(np.abs(roots)), [2.0, 2.0, 2.0], rtol=1e-12, what="the radii")
    true(bool(np.any(np.abs(roots - (-2)) < 1e-9)), "-2 is not one of the roots", "a root near -2", roots)


@check("properties", "polar form and back gives the same number", "Converting to (r, theta) and back must return the original number, in every quadrant.")
def round_trip(sub):
    z = np.array([3 + 4j, -3 + 4j, -3 - 4j, 3 - 4j, 5 + 0j, -5 + 0j, 2j, -2j])
    r, theta = sub.to_polar(z)
    cclose(sub.from_polar(r, theta), z, atol=1e-12, what="the round trip")


@check("properties", "the angle is correct in all four quadrants", "Use atan2, and not atan of the ratio, so that the quadrant is right and a real part of 0 does not divide by zero.")
def quadrants(sub):
    _, theta = sub.to_polar(np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j, 2j, -2j, -3 + 0j]))
    cclose(theta, [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2, np.pi], rtol=1e-12, what="the angles")


@check("properties", "the radius is the modulus", "The radius of x + iy is sqrt(x^2 + y^2).")
def radius(sub):
    rng = np.random.default_rng(0)
    z = rng.normal(size=20) + 1j * rng.normal(size=20)
    cclose(sub.to_polar(z)[0], np.abs(z), rtol=1e-12, what="the radius")


@check("properties", "complex_power matches repeated multiplication", "z to the power n by de Moivre equals z multiplied by itself n times.")
def power_matches(sub):
    z = 0.8 - 0.6j + 0.3
    for n in (0, 1, 2, 5, 9):
        cclose(sub.complex_power(z, n), z ** n, rtol=1e-9, atol=1e-12, what=f"the power for n={n}")


@check("properties", "roots of unity: n roots, each to the power n is 1, and they add up to 0", "The n roots of 1 are equally spaced on the unit circle. By symmetry, their sum is 0 for n > 1.")
def unity_properties(sub):
    for n in (2, 3, 5, 8):
        roots = sub.roots_of_unity(n)
        shape(roots, (n,), what=f"the roots for n={n}")
        cclose(np.abs(roots), np.ones(n), rtol=1e-12, what="the radii")
        cclose(roots ** n, np.ones(n), atol=1e-9, what="the roots to the power n")
        cclose(np.sum(roots), 0.0, atol=1e-9, what="the sum of the roots")
    cclose(sub.roots_of_unity(6)[0], 1.0, atol=1e-12, what="the first root")


@check("properties", "the n-th roots of z are n different numbers, and each to the power n gives z", "A non-zero number has exactly n different n-th roots. They are spaced by 2 pi / n on a circle.")
def nth_root_properties(sub):
    for z, n in [(-8 + 0j, 3), (1 + 1j, 4), (0.5 - 2j, 5)]:
        roots = sub.nth_roots(z, n)
        shape(roots, (n,), what="the roots")
        cclose(roots ** n, np.full(n, z), atol=1e-9, what=f"the roots of {z} to the power {n}")
        true(len({(round(v.real, 6), round(v.imag, 6)) for v in roots}) == n, "the roots are not all different", n, roots)


@check("edge", "zero", "The number 0 has radius 0 and angle 0. Its n-th roots are all 0.")
def zero(sub):
    r, theta = sub.to_polar(0j)
    cclose((r, theta), (0.0, 0.0), atol=0, what="(radius, angle)")
    cclose(sub.nth_roots(0j, 3), np.zeros(3, dtype=complex), atol=1e-12, what="the cube roots of 0")


@check("edge", "real and integer input", "A real number is a complex number with imaginary part 0.")
def real_input(sub):
    r, theta = sub.to_polar(np.array([4.0, -4.0]))
    cclose(r, [4.0, 4.0], rtol=1e-12, what="the radii")
    cclose(theta, [0.0, np.pi], rtol=1e-12, what="the angles")
    cclose(sub.to_polar(3)[0], 3.0, rtol=1e-12, what="the radius of the int 3")


@fixture("numbers")
def numbers(ref):
    rng = np.random.default_rng(42)
    for n in (2, 3, 5, 6):
        z = complex(rng.normal(), rng.normal())
        r, theta = ref.to_polar(z)
        power, roots, unity = ref.complex_power(z, n), ref.nth_roots(z, n), ref.roots_of_unity(n)
        yield {"re": z.real, "im": z.imag, "n": n}, {
            "r": float(r), "theta": float(theta),
            "power": [float(power.real), float(power.imag)],
            "roots_re": roots.real, "roots_im": roots.imag,
            "unity_re": unity.real, "unity_im": unity.imag}


@check("reference", "all functions match the reference", "Your polar form, power and roots equal the reference for random complex numbers.")
def reference_all(sub):
    for case, expected in fixtures("numbers"):
        z, n = complex(case["re"], case["im"]), case["n"]
        r, theta = sub.to_polar(z)
        cclose(r, expected["r"], rtol=1e-10, what="the radius")
        cclose(theta, expected["theta"], rtol=1e-10, what="the angle")
        cclose(sub.complex_power(z, n), complex(*expected["power"]), rtol=1e-9, atol=1e-12, what="the power")
        cclose(sub.nth_roots(z, n), np.array(expected["roots_re"]) + 1j * np.array(expected["roots_im"]), rtol=1e-9, atol=1e-12, what="the roots")
        cclose(sub.roots_of_unity(n), np.array(expected["unity_re"]) + 1j * np.array(expected["unity_im"]), rtol=1e-9, atol=1e-12, what="the roots of unity")
