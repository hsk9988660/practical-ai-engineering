Machine learning code works on batches of tensors. Most bugs come from putting a statistic on the wrong axis, and these bugs often stay silent because numpy broadcasts almost any shapes. You write four small operations that every deep learning project needs, and you use broadcasting and indexing instead of loops.

## Task

Write four functions.

1. `channel_normalize(x, mean, std)` normalizes images. `x` has the shape `(N, C, H, W)`. `mean` and `std` have one number for each channel. Return `(x - mean) / std`, where each channel uses its own numbers.
2. `one_hot(labels, n)` returns an array of shape `(len(labels), n)`. Row $i$ has a 1 at column `labels[i]` and 0 elsewhere. Raise `ValueError` if a label is below 0 or at least `n`.
3. `batched_outer(a, b)` returns, for each example $i$, the outer product of `a[i]` and `b[i]`. The result has the shape `(N, d1, d2)`.
4. `masked_mean(x, mask)` averages a sequence over time and ignores the padded steps. `x` has the shape `(N, T, D)` and `mask` has the shape `(N, T)`. A step is valid where `mask` is `True`. Return an array of shape `(N, D)` with the mean over the valid steps of each example. An example that has no valid step gets a row of zeros.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `x` (images) | float array | `(N, C, H, W)` | a batch of images |
| `mean`, `std` | float arrays | `(C,)` | one number for each channel |
| `labels` | int array | `(N,)` | the class of each example |
| `n` | int | scalar | the number of classes |
| `a`, `b` | float arrays | `(N, d1)` and `(N, d2)` | one vector of each kind for each example |
| `x` (sequences) | float array | `(N, T, D)` | a padded batch of sequences |
| `mask` | bool array | `(N, T)` | `True` for a real step and `False` for padding |

- The results have the shapes `(N, C, H, W)`, `(N, n)`, `(N, d1, d2)` and `(N, D)`.

## Constraints

- Do not use Python loops over the examples, the pixels or the time steps.
- Do not change the inputs.
- Use numpy only.

## Examples

```python
x = np.array([[[[1.0, 3.0]], [[10.0, 30.0]]]])
channel_normalize(x, np.array([2.0, 20.0]), np.array([1.0, 10.0]))
```

```text
array([[[[-1.,  1.]],

        [[-1.,  1.]]]])
```

```python
one_hot(np.array([2, 0]), 3)
```

```text
array([[0., 0., 1.],
       [1., 0., 0.]])
```

```python
batched_outer(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0, 5.0]]))
```

```text
array([[[ 3.,  4.,  5.],
        [ 6.,  8., 10.]]])
```

```python
seq = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
masked_mean(seq, np.array([[True, True, False]]))
```

```text
array([[2., 3.]])
```
