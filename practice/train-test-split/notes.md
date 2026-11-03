## Why a seed

An experiment that you cannot repeat is hard to trust. With a fixed seed, a colleague gets the same train and test sets and can compare results with yours. A seed that you do not give is a source of noise that looks like a change in the model.

## Why shuffle positions and not the list

If you shuffle the caller's list, you change the order of the caller's data. Another part of the program can depend on that order, for example when labels sit in a second list at the same positions. Shuffling a list of positions leaves the input as it was.

## Why a private generator

The global `np.random` functions share one hidden state. A library that calls them in the middle of your code changes your split. `np.random.default_rng(seed)` is a separate object that only you use.

## What this split does not do

It does not keep the class balance. With rare classes, a random split can put almost all of a class in one part. A stratified split fixes this. It also does not keep related items together. If two items are copies or come from the same person, split by group.

## Where this goes next

The classical ML phase adds a validation set and cross-validation. The same rules apply: split first, fit scalers and models on the train part only, and use the test part once at the end.
