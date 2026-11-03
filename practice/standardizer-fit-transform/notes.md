## Fit on train, apply to everything

The mean and the scale are parameters that you learn, like the weights of a model. Learn them from the train data. Then apply the same numbers to the validation data, the test data and the data that the model sees in production. A test set that is standardized with its own statistics hides a shift between train and test, and a model that works in the test can fail in use.

## Why two functions

Separate `fit` and `apply` steps make this rule easy to follow. You can save the pair `(mean, scale)` with the model and load it later. This is the idea of `StandardScaler` in scikit-learn.

## The constant column

A feature that never changes has no information, and its standard deviation is 0. A division by 0 gives NaN, and one NaN can spoil a whole training run. A scale of 1.0 keeps the column at 0 after the centering.

## Population or sample

This task uses the population standard deviation, as scikit-learn does. For a large `n` the difference to the sample standard deviation is small. For a small `n` it is not, and a mix of the two conventions gives results that do not match.

## Where this goes next

Batch normalization in the deep learning core phase does the same operation inside the network, on each batch, with a learned scale and shift. Layer normalization does it over the features of each example.
