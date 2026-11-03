A model must be tested on data that it did not train on. You split the data into a train part and a test part before training. The split must be random, but you must be able to repeat it, so that an experiment gives the same result when you run it again.

## Task

Write `train_test_split(items, test_fraction=0.2, seed=0)`. It returns `(train, test)`, two lists.

1. Raise `ValueError` if `test_fraction` is not strictly between 0 and 1.
2. Raise `ValueError` if `items` has fewer than 2 items.
3. Shuffle the positions of the items with `np.random.default_rng(seed)`. The same `seed` must always give the same split.
4. The test part has `round(len(items) * test_fraction)` items. Keep this number between 1 and `len(items) - 1`, so that both parts are never empty.
5. Every item is in exactly one part.

## Input and output

| Name | Type | Meaning |
|---|---|---|
| `items` | list | the data. Items can be any value |
| `test_fraction` | float | the share of items that go to the test part |
| `seed` | int | the seed of the random generator |

- The function returns a tuple of two new lists: `(train, test)`.

## Constraints

- Use numpy only.
- Do not change `items`.
- Do not use the global `np.random` functions. Make your own generator from `seed`.

## Examples

```python
train, test = train_test_split(list(range(10)), 0.3, seed=1)
len(train), len(test)
sorted(train + test)
train_test_split(list(range(10)), 0.3, seed=1) == train_test_split(list(range(10)), 0.3, seed=1)
len(train_test_split(list("abcde"), 0.01)[1])
```

```text
(7, 3)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
True
1
```
