## Why broadcasting bugs are silent

Numpy adds missing axes on the left and stretches axes of size 1. A `(C,)` array can be combined with almost any tensor, and the code runs without an error. When `C` equals `W` the result is wrong but has the right shape. The tests use channel and width sizes that differ, so a wrong axis shows up.

## Negative indices

`out[i, -1] = 1.0` sets the last column. A label of -1 from a bad data pipeline would train the model on the wrong class without any warning. A range check costs little.

## Padding

Batches of sequences have different lengths, so shorter ones are padded. A mean over all steps would include the padding and bias the result. The mask removes it. The same pattern appears in attention masks and in loss masks for language models.

## Loops and speed

The operations here run in compiled code. A Python loop over a batch is often 100 times slower. Write the loop-free version, and use a loop version only as a test oracle, as the tests do.

## Where this goes next

Per-channel normalization is the first step of most vision pipelines. One-hot labels feed the cross-entropy. Masked means appear in pooling over tokens, and batched outer products appear in attention and in LoRA updates.
