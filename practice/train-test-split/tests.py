import numpy as np

from practice_check import check, fixture, fixtures, raises, true


def same(actual, expected, what):
    true(actual == expected, f"{what} is not as expected", expected, actual)


@check("examples", "sizes for 10 items and fraction 0.3", "round(10 * 0.3) is 3 test items, so the train part has 7.")
def example_sizes(sub):
    train, test = sub.train_test_split(list(range(10)), 0.3, seed=1)
    same((len(train), len(test)), (7, 3), "the part sizes (train, test)")


@check("examples", "every item is in exactly one part", "The two parts together hold each input item one time.")
def example_partition(sub):
    train, test = sub.train_test_split(list(range(10)), 0.3, seed=1)
    same(sorted(train + test), list(range(10)), "the union of the parts")


@check("examples", "the same seed gives the same split", "The split must be repeatable. Use a generator made from the seed.")
def example_repeat(sub):
    data = list(range(10))
    same(sub.train_test_split(data, 0.3, seed=1), sub.train_test_split(data, 0.3, seed=1), "two splits with seed 1")


@check("examples", "a tiny fraction still gives one test item", "round(5 * 0.01) is 0, but the test part must have at least 1 item.")
def example_clamp_low(sub):
    same(len(sub.train_test_split(list("abcde"), 0.01)[1]), 1, "the test size")


@check("properties", "the parts do not overlap", "An item in both parts would leak test data into training. Items that are all different must appear once.")
def no_overlap(sub):
    train, test = sub.train_test_split(list(range(50)), 0.25, seed=3)
    true(not set(train) & set(test), "an item is in both parts", "no common items", sorted(set(train) & set(test)))


@check("properties", "different seeds give different splits", "A seed picks one shuffle. Two seeds should not give the same test set for a list of 50 items.")
def seeds_differ(sub):
    data = list(range(50))
    tests = [tuple(sub.train_test_split(data, 0.2, seed=s)[1]) for s in range(5)]
    true(len(set(tests)) > 1, "five seeds gave the same test set", "at least 2 different test sets", tests[0])


@check("properties", "the test set is a random sample, not the first items", "Without a shuffle, the test set is the first or last items of the list.")
def is_shuffled(sub):
    data = list(range(100))
    for seed in range(3):
        test = sub.train_test_split(data, 0.2, seed=seed)[1]
        true(test != data[:20] and test != data[-20:], "the test set is a slice of the input", "a shuffled sample", test[:5])


@check("properties", "the input list is not changed", "Shuffling the caller's list changes its order. Shuffle a list of positions instead.")
def input_unchanged(sub):
    data = list(range(30))
    sub.train_test_split(data, 0.3, seed=2)
    same(data, list(range(30)), "the input list")


@check("properties", "the function returns the train part first", "The result is (train, test). The larger part comes first when the fraction is below 0.5.")
def train_first(sub):
    train, test = sub.train_test_split(list(range(10)), 0.2, seed=0)
    same((len(train), len(test)), (8, 2), "the part sizes (train, test)")


@check("properties", "the global numpy generator is not used", "Use np.random.default_rng(seed). The state of the global generator must stay as it was.")
def global_state(sub):
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    sub.train_test_split(list(range(20)), 0.3, seed=5)
    after = np.random.get_state()[1]
    true(bool(np.array_equal(before, after)), "the call changed the state of the global numpy generator", "the same state", "a different state")


@check("edge", "two items give one train item and one test item", "With 2 items both parts must be non-empty, whatever the fraction is.")
def two_items(sub):
    for fraction in (0.01, 0.5, 0.99):
        train, test = sub.train_test_split([10, 20], fraction, seed=0)
        same((len(train), len(test)), (1, 1), f"the sizes for fraction {fraction}")


@check("edge", "a large fraction keeps one train item", "round(5 * 0.99) is 5, but the train part must have at least 1 item.")
def clamp_high(sub):
    train, test = sub.train_test_split(list("abcde"), 0.99)
    same((len(train), len(test)), (1, 4), "the part sizes (train, test)")


@check("edge", "bad arguments raise ValueError", "The fraction must be strictly between 0 and 1, and the list needs at least 2 items.")
def bad_args(sub):
    raises(ValueError, sub.train_test_split, [1, 2, 3], 0.0)
    raises(ValueError, sub.train_test_split, [1, 2, 3], 1.0)
    raises(ValueError, sub.train_test_split, [1, 2, 3], -0.5)
    raises(ValueError, sub.train_test_split, [1], 0.5)
    raises(ValueError, sub.train_test_split, [], 0.5)


@check("edge", "items can repeat and can be any type", "The function works on positions, so equal items stay as they are and are not merged.")
def repeated_items(sub):
    data = ["a", "a", "b", "b", "b", None]
    train, test = sub.train_test_split(data, 0.5, seed=4)
    same(sorted(map(str, train + test)), sorted(map(str, data)), "the union of the parts")


@fixture("sizes")
def sizes(ref):
    for n, fraction in [(2, 0.5), (3, 0.5), (5, 0.01), (5, 0.99), (10, 0.3), (10, 0.25), (17, 0.2), (100, 0.15), (101, 0.5), (1000, 0.1)]:
        train, test = ref.train_test_split(list(range(n)), fraction, seed=7)
        yield {"n": n, "fraction": fraction}, [len(train), len(test)]


@check("reference", "part sizes match the reference", "Your test size uses round() and the clamp, and agrees with the reference for many n and fractions.")
def reference_sizes(sub):
    for case, expected in fixtures("sizes"):
        train, test = sub.train_test_split(list(range(case["n"])), case["fraction"], seed=7)
        same([len(train), len(test)], expected, f"the sizes for n={case['n']}, fraction={case['fraction']}")
