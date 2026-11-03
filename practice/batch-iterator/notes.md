## Why lazy

A list of one million images does not fit in memory, but a generator that reads one file at a time does. If `batches` copied its input into a list, the generator would lose its benefit. Taking `batch_size` items at a time keeps memory use at one batch.

## Why a new list for each batch

If a function reuses one list, a caller that keeps the batches sees them all change. Training loops often keep a few batches, for example for prefetching. A new list for each batch is the safe choice.

## Why check the argument in the outer function

A generator function does not run any of its code until the first `next`. A bad `batch_size` would then fail late and far from the cause. The outer function checks first and returns the generator after that.

## Where this goes next

PyTorch `DataLoader` and the data loaders in the later phases do the same job and add shuffling, workers and padding. The `drop_last` option matters when a layer such as batch normalization needs a full batch.
