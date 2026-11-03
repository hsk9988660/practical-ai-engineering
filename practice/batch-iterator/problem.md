Training code reads data in batches. A dataset can be too large for memory, so the batching code must read items only when it needs them. You write a function that turns any iterable into an iterator of fixed-size lists.

## Task

Write `batches(items, batch_size, drop_last=False)`.

- It returns an iterator. Each item of the iterator is a list of `batch_size` items from `items`, in the same order.
- The last batch can be shorter. When `drop_last` is `True`, the function does not return a last batch that is shorter than `batch_size`.
- The function reads `items` lazily. It takes only the items that the next batch needs.
- If `batch_size` is less than 1, the function raises `ValueError`.

## Input and output

| Name | Type | Meaning |
|---|---|---|
| `items` | any iterable | a list, a range or a generator. A generator can be read only once |
| `batch_size` | int | the number of items in a full batch |
| `drop_last` | bool | `True` removes a last batch that is too short |

- `batches` returns an iterator. A call to `next` on it gives one list.

## Constraints

- Use the standard library only.
- Do not change `items` when it is a list.
- Each batch must be a new list. A later step of the iterator must not change a batch that you already gave out.

## Examples

```python
list(batches(range(7), 3))
list(batches(range(7), 3, drop_last=True))
list(batches([], 3))
```

```text
[[0, 1, 2], [3, 4, 5], [6]]
[[0, 1, 2], [3, 4, 5]]
[]
```
