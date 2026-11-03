## Hint 1

Do the two checks first and raise `ValueError` with a clear message. Then compute the number of test items.

## Hint 2

Do not shuffle `items`. Shuffle a list of positions, and use the shuffled positions to pick the items.

## Hint 3

Make a generator with `np.random.default_rng(seed)` and call its `permutation` method with `len(items)`. A generator with the same seed gives the same permutation every time.

## Hint 4

Clamp the test size with `min(max(1, n_test), len(items) - 1)`. Then the first `n_test` shuffled positions give the test items and the rest give the train items.
