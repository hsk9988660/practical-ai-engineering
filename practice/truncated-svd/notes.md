## Why this is the best approximation

The Eckart-Young theorem says that no matrix of rank $k$ is closer to $A$ than the truncated SVD, in the Frobenius norm and in the spectral norm. The Frobenius error is exactly the square root of the sum of the squared singular values that you drop. The tests check this.

## Choosing k

A plot of the cumulative energy shows how fast the singular values fall. When they fall fast, a small `k` keeps almost everything. Real data such as images, user ratings and weight matrices of neural networks often has this shape.

## Storage

A matrix of size $m \times n$ needs $mn$ numbers. A rank $k$ approximation needs $k(m + n + 1)$ numbers in factored form. For $k$ much smaller than $m$ and $n$, this is a large saving.

## Where this goes next

LoRA, in the LLM engineering phase, adds a trained low-rank update to a frozen weight matrix. PCA is the SVD of centered data. Latent semantic analysis and many recommender systems use the same truncation.
