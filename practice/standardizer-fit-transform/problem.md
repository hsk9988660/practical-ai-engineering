Many models train better when every feature has a mean of 0 and a standard deviation of 1. You learn the mean and the scale from the train data only, and then you use the same numbers on the test data. If you compute them again on the test data, information from the test set leaks into your pipeline.

## Task

Write two functions.

1. `fit_standardizer(x)` returns `(mean, scale)`, computed for each column of `x`.
   - `mean` is the mean of the column.
   - `scale` is the population standard deviation of the column (divide by $n$, not by $n - 1$). If the standard deviation of a column is 0, its scale is `1.0`.
2. `apply_standardizer(x, mean, scale)` returns a new array with

$$\frac{x - \text{mean}}{\text{scale}}$$

for every row of `x`.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `x` | float array | `(n_rows, n_features)` | the data |
| `mean` | float array | `(n_features,)` | the column means from the train data |
| `scale` | float array | `(n_features,)` | the column scales from the train data. No value is 0 |

- `fit_standardizer` returns a tuple of two arrays with the shape `(n_features,)`.
- `apply_standardizer` returns a float array with the shape of `x`.

## Constraints

- `x` has at least 1 row.
- Do not change `x`, `mean` or `scale`.
- Use numpy only.

## Examples

```python
x = np.array([[1.0, 10.0, 5.0], [3.0, 20.0, 5.0], [5.0, 30.0, 5.0]])
mean, scale = fit_standardizer(x)
mean
scale
apply_standardizer(x, mean, scale)
apply_standardizer(np.array([[7.0, 25.0, 6.0]]), mean, scale)
```

```text
array([ 3., 20.,  5.])
array([1.63299316, 8.16496581, 1.        ])
array([[-1.22474487, -1.22474487,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 1.22474487,  1.22474487,  0.        ]])
array([[2.44948974, 0.61237244, 1.        ]])
```
