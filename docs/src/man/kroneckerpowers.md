# Kronecker powers and graphs

## Kronecker powers

Repeated Kronecker multiplications of the same matrix, i.e.

```math
A^{\otimes n}=\otimes_{i=1}^n A = \underbrace{A \otimes A \otimes \ldots \otimes A}_{n\text{ times}}\,.
```

Kronecker powers are supported using `kronecker(A, n)` or, equivalently, `⊗(A, n)`. These functions yield an instance of `KroneckerPower`, as struct which holds the matrix and the power. It works just like instances of `KroneckerProduct`, but more efficient since only a single matrix has to be stored and manipulated. These products work as expected.

```@repl
using Kronecker
A = rand(2, 2)
K2 = A ⊗ 2
inv(K2)
K12 = K2 ⊗ 6  # works recursively
K12^2  # example
```

```@docs
kronecker(A::AbstractMatrix, pow::Int)
⊗(A::AbstractMatrix, pow::Int)
```

## Kronecker graphs

An exciting application of Kronecker powers (or Krocker products in general) is generating large, realistic graphs from an initial 'seed' via a stochastic process. This is called a Kronecker graph and is described in detail in [Leskovec et al. (2008)](https://cs.stanford.edu/~jure/pubs/kronecker-jmlr10.pdf).

If we use an initial matrix $P$ with values in $[0,1]$ as a seed, then $P_n=P^{\otimes n}$ can be seen as a probability distribution for an adjacency matrix of a graph. The elements $(P_n)_{i,j}$ give the probabilities that there is an edge from node $i$ to node $j$. Leskovec and co-authors give two algorithms for sampling adjacency matrices from this distribution, both provided by `Kronecker.jl`:

- `naivesample` gives exact samples, but has a computational time proportional to the number of elements and hence prohibitive for large graphs;
- `fastsample` a recursive heuristic, has a computational time proportional to the expected number of edges in the graph.

The latter can easily scale to generate graphs of millions of nodes. Both return the adjacency matrix as a sparse array.

```@repl
using Kronecker
P = rand(2, 2)
P10 = P ⊗ 10
sum(P10)  # expected number of edges
A = fastsample(P10)
```

```@docs
isprob
naivesample
fastsample
```
