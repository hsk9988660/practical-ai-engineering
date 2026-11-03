## What PCA does

The first component is the direction in which the centered data has the largest variance. The second is the direction with the next largest variance that is at right angles to the first, and so on. The eigenvalue of a component is the variance of the data along it. This is why the ratio of an eigenvalue to the sum of all of them is the share of the variance that the component explains.

## Why the data must be centered

Without centering, the covariance matrix is not a covariance matrix. If the data is far from the origin, the first "component" points at the mean and not along the spread of the data.

## Sign and order

Eigenvectors have no fixed sign, and tools return different signs. A sign rule makes your results repeatable and comparable between runs and machines. The order is not free either: `eigh` returns the smallest eigenvalue first, but you want the largest first.

## Reconstruction

If you keep all $d$ components, the reconstruction is exact. With $k < d$ you lose the part of the data along the dropped components. The mean squared reconstruction error is the sum of the dropped eigenvalues, up to the factor $(n-1)/n$. This is why the cumulative ratio is used to choose $k$.

## Scale

PCA is sensitive to the scale of the features. A feature in millimeters dominates a feature in meters. Standardize the columns first when the units differ, as in the standardizer exercise.

## Where this goes next

SVD gives the same components without the covariance matrix, and is the stable way to do it for a large number of features. PCA is also used for visualization, for noise removal, and as a first step before other methods such as clustering.
