import numpy as np

from practice_check import check, close, equal, fixture, fixtures, raises, true

P = np.array([0.5, 0.5])
Q = np.array([0.9, 0.1])


@check("examples", "entropy of a fair coin and of a sure outcome", "A fair coin has 1 bit. An outcome with probability 1 has 0 bits, and the result must be 0.0 and not -0.0.")
def example_entropy(sub):
    close(sub.entropy(P), 1.0, atol=1e-12, what="entropy of the fair coin")
    equal(str(sub.entropy(np.array([1.0, 0.0]))), "0.0", what="entropy of a sure outcome")


@check("examples", "entropy of the uniform distribution over 4 outcomes", "The uniform distribution over n outcomes has entropy log2(n), which is 2 bits for n = 4.")
def example_uniform(sub):
    close(sub.entropy(np.full(4, 0.25)), 2.0, atol=1e-12, what="entropy")


@check("examples", "entropy in nats", "With base e, a fair coin has ln 2 = 0.6931... nats.")
def example_nats(sub):
    close(sub.entropy(P, base=np.e), 0.6931471805599453, rtol=1e-12, what="entropy in nats")


@check("examples", "cross-entropy and KL divergence of two coins", "H(p, q) = -0.5 log2 0.9 - 0.5 log2 0.1, and KL is H(p, q) - H(p).")
def example_ce_kl(sub):
    close(sub.cross_entropy(P, Q), 1.736965594166206, rtol=1e-12, what="cross-entropy")
    close(sub.kl_divergence(P, Q), 0.7369655941662061, rtol=1e-12, what="KL divergence")


@check("examples", "KL divergence is infinite when q is 0 and p is not", "The distribution q says that outcome 1 is impossible, but p can produce it.")
def example_inf(sub):
    equal(sub.kl_divergence(P, np.array([1.0, 0.0])), float("inf"), what="KL divergence")


@check("properties", "KL divergence is never negative and is 0 only for equal distributions", "Gibbs' inequality: D(p || q) >= 0, with equality when p = q.")
def kl_non_negative(sub):
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = rng.random(5)
        p /= p.sum()
        q = rng.random(5)
        q /= q.sum()
        true(sub.kl_divergence(p, q) >= -1e-12, "KL divergence is negative", "0 or more", sub.kl_divergence(p, q))
        close(sub.kl_divergence(p, p), 0.0, atol=1e-12, what="KL divergence of a distribution with itself")


@check("properties", "KL divergence is not symmetric", "D(p || q) and D(q || p) are different in general. The direction matters.")
def kl_asymmetric(sub):
    forward = sub.kl_divergence(np.array([0.9, 0.1]), np.array([0.5, 0.5]))
    backward = sub.kl_divergence(np.array([0.5, 0.5]), np.array([0.9, 0.1]))
    close(forward, 0.5310044064107188, rtol=1e-9, what="D(p || q)")
    close(backward, 0.7369655941662061, rtol=1e-9, what="D(q || p)")


@check("properties", "cross-entropy is entropy plus KL divergence", "H(p, q) = H(p) + D(p || q).")
def decomposition(sub):
    rng = np.random.default_rng(1)
    p = rng.random(6)
    p /= p.sum()
    q = rng.random(6)
    q /= q.sum()
    close(sub.cross_entropy(p, q), sub.entropy(p) + sub.kl_divergence(p, q), rtol=1e-12, what="the cross-entropy")


@check("properties", "the uniform distribution has the largest entropy", "No distribution over 5 outcomes has an entropy above log2(5).")
def uniform_is_max(sub):
    rng = np.random.default_rng(2)
    for _ in range(10):
        p = rng.random(5)
        p /= p.sum()
        true(sub.entropy(p) <= np.log2(5) + 1e-12, "the entropy is above log2(5)", "at most log2(5)", sub.entropy(p))


@check("properties", "the base changes the unit only", "Entropy in bits equals entropy in nats divided by ln 2.")
def base_change(sub):
    p = np.array([0.2, 0.3, 0.5])
    close(sub.entropy(p, 2), sub.entropy(p, np.e) / np.log(2), rtol=1e-12, what="entropy in bits")
    close(sub.entropy(p, 10), sub.entropy(p, np.e) / np.log(10), rtol=1e-12, what="entropy in dits")


@check("properties", "the inputs are not changed", "A function that writes into its input arrays breaks the code that calls it.")
def inputs_unchanged(sub):
    p, q = P.copy(), Q.copy()
    sub.entropy(p)
    sub.cross_entropy(p, q)
    sub.kl_divergence(p, q)
    close(p, P, atol=0, rtol=0, what="p")
    close(q, Q, atol=0, rtol=0, what="q")


@check("edge", "zero probabilities in p are skipped", "A term 0 * log 0 counts as 0, also when q is 0 at the same place. The result must not be NaN.")
def zeros_in_p(sub):
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.25, 0.75, 0.0])
    close(sub.entropy(p), 1.0, atol=1e-12, what="entropy")
    close(sub.cross_entropy(p, q), 0.5 * 2 + 0.5 * -np.log2(0.75), rtol=1e-12, what="cross-entropy")
    true(np.isfinite(sub.kl_divergence(p, q)), "KL divergence is not finite", "a finite number", sub.kl_divergence(p, q))


@check("edge", "zero in q where p is positive gives infinity", "Both the cross-entropy and the KL divergence are infinite.")
def inf_cases(sub):
    p = np.array([0.7, 0.3])
    q = np.array([1.0, 0.0])
    equal(sub.cross_entropy(p, q), float("inf"), what="cross-entropy")
    equal(sub.kl_divergence(p, q), float("inf"), what="KL divergence")


@check("edge", "invalid distributions raise ValueError", "A negative entry, or a sum that is not 1, is not a distribution.")
def invalid(sub):
    raises(ValueError, sub.entropy, np.array([0.5, 0.6]))
    raises(ValueError, sub.entropy, np.array([1.5, -0.5]))
    raises(ValueError, sub.cross_entropy, np.array([0.5, 0.5]), np.array([0.2, 0.2]))


@fixture("pairs")
def pairs(ref):
    rng = np.random.default_rng(42)
    for n in (2, 3, 5, 10):
        p = rng.random(n)
        p /= p.sum()
        q = rng.random(n)
        q /= q.sum()
        p[0] = 0.0
        p /= p.sum()
        yield {"p": p, "q": q}, {"h": ref.entropy(p), "ce": ref.cross_entropy(p, q), "kl": ref.kl_divergence(p, q), "h_nats": ref.entropy(p, np.e)}


@check("reference", "all three match the reference", "Your entropy, cross-entropy and KL divergence agree with the reference on random distributions that contain a zero.")
def reference_all(sub):
    for case, expected in fixtures("pairs"):
        p, q = np.array(case["p"]), np.array(case["q"])
        close(sub.entropy(p), expected["h"], rtol=1e-10, what="entropy")
        close(sub.entropy(p, np.e), expected["h_nats"], rtol=1e-10, what="entropy in nats")
        close(sub.cross_entropy(p, q), expected["ce"], rtol=1e-10, what="cross-entropy")
        close(sub.kl_divergence(p, q), expected["kl"], rtol=1e-10, what="KL divergence")
