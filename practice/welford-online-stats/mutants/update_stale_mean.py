def welford_update(state, x):
    n, mean, m2 = state
    n += 1
    delta = x - mean
    mean += delta / n
    m2 += delta * delta
    return (n, mean, m2)


def welford_merge(a, b):
    na, mean_a, m2_a = a
    nb, mean_b, m2_b = b
    n = na + nb
    if n == 0:
        return (0, 0.0, 0.0)
    delta = mean_b - mean_a
    mean = mean_a + delta * nb / n
    m2 = m2_a + m2_b + delta ** 2 * na * nb / n
    return (n, mean, m2)


def welford_finalize(state, ddof=0):
    n, mean, m2 = state
    if n - ddof <= 0:
        raise ValueError("not enough values")
    return mean, m2 / (n - ddof)
