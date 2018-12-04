# Kronecker.jl

This is a Julia package to efficiently work with Kronecker products. It combines lazy evaluation and algebraic tricks such that it can implicitely work with huge matrices. 

## Features

Given two two matrices (subtype of `AbstractArray`) `A` and `B`, one can construct an instance of the `KroneckerProduct` type as `K = A ⊗ B` (typed using `\otimes TAB`). Several functions are implemented.

- `collect(K)` computes the Kronecker product (**not** recommended!)
- `tr`, `det`, `size`, `eltype`, `inv` ... are efficient functions to work with Kronecker products
- multiplying with a vector `v` is efficient using the [vec trick](https://en.wikipedia.org/wiki/Kronecker_product#Matrix_equations): `K * v`
- solving systems of the form `A ⊗ B + D`, with `D` a diagonal matrix
- working with incomplete systems using the [sampled vec trick](https://arxiv.org/pdf/1601.01507.pdf)
- [in progress] GPU compatibility!
- [in progress] autodiff for machine learning models!

## Installation

Currently only directly from the repo. In Julia package manager (in REPL start with `]`):

```julialang
add https://github.com/MichielStock/Kronecker.jl
```

## Issues

This is very much a work in progress! Please start an issue for bugs or requests to improve functionality. Any feedback is appreciated!
