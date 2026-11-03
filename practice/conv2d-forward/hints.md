## Hint 1

First pad the image with `np.pad`. Pad only the two spatial axes, not the channel axis.

## Hint 2

Use `output_size` to get the number of rows and columns of the output. Make an array of zeros with that shape and fill it in two loops over $i$ and $j$.

## Hint 3

For each position, cut a patch from the padded image. It has all input channels and spans `kh` rows and `kw` columns. The start row is `i * stride` and the start column is `j * stride`.

## Hint 4

The patch has the shape `(C_in, kh, kw)` and `w` has the shape `(C_out, C_in, kh, kw)`. Multiply them element by element and sum over the last three axes. Do this once for all output channels with `np.tensordot` or `np.einsum`. Add the bias at the end.
