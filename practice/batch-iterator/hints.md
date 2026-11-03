## Hint 1

Check `batch_size` first and raise `ValueError` when it is less than 1. Do this in the function that the learner calls, so the error comes at the call and not at the first `next`.

## Hint 2

Call `iter(items)` once. A list and a generator both give an iterator, and the iterator remembers its position.

## Hint 3

The function `itertools.islice(iterator, n)` takes at most `n` items from an iterator and leaves the rest. Wrap it in `list(...)` to get one batch.

## Hint 4

Loop until a batch is empty. With `drop_last`, also stop when a batch has fewer than `batch_size` items. Write the loop as a generator with `yield`.
