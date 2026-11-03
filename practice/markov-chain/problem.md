A Markov chain moves between states. The next state depends only on the current state, with probabilities that are given by a transition matrix. Chains model text generation with a fixed context, the web surfer of PageRank, and the environment of reinforcement learning. Three questions matter: where is the chain after $n$ steps, where does it spend its time in the long run, and what does one run look like?

## Task

A transition matrix `p` has the shape `(k, k)`. `p[i, j]` is the probability to go from state `i` to state `j`, so every row adds up to 1. A state distribution is a row vector of length `k`.

Write three functions. All three raise `ValueError` if `p` has a negative entry or a row that does not add up to 1 (use a tolerance of `1e-8`).

1. `n_step(p0, p, n)` returns the distribution after `n` steps from the distribution `p0`. One step multiplies the row vector by the matrix: $\pi \leftarrow \pi P$. With `n = 0` it returns `p0`.
2. `stationary_distribution(p)` returns the distribution $\pi$ with $\pi P = \pi$ and $\sum_i \pi_i = 1$. Solve this linear system: take the system $(P^\top - I)\pi = 0$, replace its **last row** with a row of ones, and set the last entry of the right-hand side to 1. Assume that the chain has a single stationary distribution.
3. `simulate(p, start, uniforms)` returns the list of visited states: first `start`, then one state for each number in `uniforms`. From state $s$, the next state is chosen like in a categorical sample: the first state whose cumulative probability in row $s$ is **greater than** the uniform number $u$. A $u$ that equals a cumulative value belongs to the next state.

## Input and output

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `p` | float array | `(k, k)` | the transition matrix |
| `p0` | float array | `(k,)` | the start distribution |
| `n` | int | scalar | the number of steps, at least 0 |
| `start` | int | scalar | the first state |
| `uniforms` | list of float | `(T,)` | one uniform number in $[0, 1)$ for each step |

- `n_step` and `stationary_distribution` return a float array with the shape `(k,)`.
- `simulate` returns a list of `T + 1` ints.

## Constraints

- Do not change the inputs.
- Do not use `np.random`. The random numbers are inputs.
- Use numpy only.

## Examples

```python
p = np.array([[0.9, 0.1], [0.5, 0.5]])
n_step(np.array([1.0, 0.0]), p, 1)
n_step(np.array([1.0, 0.0]), p, 2)
stationary_distribution(p)
simulate(p, 0, [0.95, 0.3, 0.6])
```

```text
array([0.9, 0.1])
array([0.86, 0.14])
array([0.83333333, 0.16666667])
[0, 1, 0, 0]
```
