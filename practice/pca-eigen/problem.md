Principal component analysis (PCA) finds the directions along which the data varies most. You keep the first few directions and drop the rest. This compresses the data and often removes noise.

## Task

Write three functions.

1. `fit_pca(x, k)` returns `(mean, components, ratio)`.
   1. `mean` is the mean of each column of `x`. Subtract it from `x`.
   2. Build the covariance matrix $C = \frac{1}{n-1} X_c^\top X_c$ of the centered data $X_c$.
   3. Compute the eigenvalues and eigenvectors of $C$ with `np.linalg.eigh`. Sort them by eigenvalue, **largest first**. Set negative eigenvalues that come from rounding to 0.
   4. Fix the sign of every eigenvector: the entry with the largest absolute value must be positive.
   5. `components` has the first `k` eigenvectors as its **rows**. `ratio` has the share of the total variance of each of the first `k` components: eigenvalue divided by the sum of **all** eigenvalues.
2. `transform(x, mean, components)` returns the coordinates of the data along the components: $(x - \text{mean})\,W^\top$.
3. `inverse_transform(z, mean, components)` returns $z W + \text{mean}$, the reconstruction in the original space.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `x` | float array | `(n, d)` | the data, one sample in each row, with $n \ge 2$ |
| `k` | int | scalar | the number of components to keep, $1 \le k \le d$ |
| `mean` | float array | `(d,)` | the mean of the training data |
| `components` | float array | `(k, d)` | the principal axes, one in each row |
| `z` | float array | `(n, k)` | the coordinates in the new space |

- `fit_pca` returns `mean` with the shape `(d,)`, `components` with the shape `(k, d)`, and `ratio` with the shape `(k,)`.
- `transform` returns the shape `(n, k)`. `inverse_transform` returns the shape `(n, d)`.

## Constraints

- Do not change `x`.
- Use numpy only. You can use `np.linalg.eigh`, but not a ready-made PCA.

## Examples

```python
x = np.array([[1.0, 2.0], [2.0, 4.1], [3.0, 5.9], [4.0, 8.2]])
mean, components, ratio = fit_pca(x, 1)
mean
components
ratio
z = transform(x, mean, components)
z.ravel()
inverse_transform(z, mean, components)
```

```text
array([2.5 , 5.05])
array([[0.43958017, 0.89820336]])
array([0.99968563])
array([-3.3988905 , -1.07308328,  0.98326294,  3.48871084])
array([[1.00591515, 1.99710513],
       [2.02829388, 4.08615299],
       [2.93222289, 5.93317008],
       [4.03356809, 8.1835718 ]])
```
