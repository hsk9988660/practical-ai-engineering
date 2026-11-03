A convolution layer slides a small kernel over an image and takes a weighted sum at every position. It is the main building block of a CNN. You write the forward pass for one image with many channels.

## Task

Write two functions.

1. `output_size(size, kernel, padding, stride)` returns the number of output positions along one axis:

$$\left\lfloor \frac{\text{size} + 2\,\text{padding} - \text{kernel}}{\text{stride}} \right\rfloor + 1$$

2. `conv2d(x, w, b=None, stride=1, padding=0)` returns the output of the layer. For output channel $o$ and position $(i, j)$:

$$\text{out}[o, i, j] = b[o] + \sum_{c}\sum_{u}\sum_{v} w[o, c, u, v] \; \tilde{x}[c,\; i \cdot s + u,\; j \cdot s + v]$$

Here $\tilde{x}$ is $x$ after zero padding on all four sides, and $s$ is the stride. The kernel is not flipped. This is cross-correlation, which is what deep learning libraries call convolution.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `x` | float array | `(C_in, H, W)` | one image |
| `w` | float array | `(C_out, C_in, kh, kw)` | the kernels |
| `b` | float array or `None` | `(C_out,)` | the bias, one number for each output channel |
| `stride` | int | scalar | the step of the window, at least 1 |
| `padding` | int | scalar | the number of zero rows and columns added on each side |

- `output_size` returns an int.
- `conv2d` returns a float array with the shape `(C_out, H_out, W_out)`. `H_out` and `W_out` come from `output_size`.

## Constraints

- Assume that the kernel fits in the padded image.
- Do not change `x`, `w` or `b`.
- Use numpy only.

## Examples

```python
x = np.arange(16, dtype=float).reshape(1, 4, 4)
w = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
conv2d(x, w)
conv2d(x, w, stride=2)
conv2d(x, w, b=np.array([0.5]), padding=1)
```

```text
array([[[-5., -5., -5.],
        [-5., -5., -5.],
        [-5., -5., -5.]]])
array([[[-5., -5.],
        [-5., -5.]]])
array([[[  0.5,  -0.5,  -1.5,  -2.5,   0.5],
        [ -3.5,  -4.5,  -4.5,  -4.5,   3.5],
        [ -7.5,  -4.5,  -4.5,  -4.5,   7.5],
        [-11.5,  -4.5,  -4.5,  -4.5,  11.5],
        [  0.5,  12.5,  13.5,  14.5,  15.5]]])
```

```python
output_size(32, 3, 1, 1)
output_size(7, 3, 0, 2)
```

```text
32
3
```
