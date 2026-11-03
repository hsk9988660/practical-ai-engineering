## Rows and columns

In this convention, rows are the current state and columns the next state, and distributions are row vectors. Many textbooks and libraries use the other convention, with column vectors and a transposed matrix. Mixing the two gives wrong answers that still look like probabilities, so always check which one you use.

## The stationary distribution

A stationary distribution is a left eigenvector of $P$ for the eigenvalue 1. For a chain in which every state can reach every other state, it is unique, and the long-run share of time in each state converges to it. The linear system in this exercise is a direct way to get it. For large chains, power iteration is used: apply the matrix many times.

## Convergence

The distribution after $n$ steps approaches the stationary distribution at a speed that depends on the second largest eigenvalue of $P$. A chain that is almost split into two parts mixes slowly. A chain with an absorbing state ends there with probability 1.

## Simulation

Taking the uniform numbers as input makes a run exactly repeatable, and the check is exact. The choice of the next state is the categorical sampling of the sampling exercise, applied to the row of the current state.

## Where this goes next

PageRank is the stationary distribution of the random surfer chain. Markov decision processes in the reinforcement learning phase add actions and rewards to the chain. Markov chain Monte Carlo builds a chain whose stationary distribution is the distribution that you want to sample from.
