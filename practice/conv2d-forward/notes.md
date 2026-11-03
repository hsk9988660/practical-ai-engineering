## Cross-correlation, not convolution

A mathematical convolution flips the kernel. A deep learning layer does not. The network learns the kernel, so a flipped kernel would learn the same function with flipped weights. Frameworks skip the flip and keep the name.

## What the output size means

The window must fit inside the padded image. With `padding = (kernel - 1) // 2` and `stride = 1`, the output has the same size as the input. The floor division drops the last positions when the window does not fit exactly. This is why stride 2 on an axis of size 7 with kernel 3 gives 3 outputs.

## Cost

Each output value needs `C_in * kh * kw` multiplications. The loop version is slow because Python runs the position loop. The lesson shows the im2col method, which turns the whole layer into one matrix multiplication. Real libraries use that idea, or an FFT or Winograd method.

## Where this goes next

The backward pass of this layer is another convolution, with the kernel flipped and the gradient as the input. Pooling, strides and padding appear again in the CNN architectures lesson and in the detection and segmentation lessons.
