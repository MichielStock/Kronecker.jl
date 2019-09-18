# Kronecker.jl

*A general-purpose toolbox for efficient Kronecker-based algebra.*

`Kronecker.jl` is a Julia package for working with large-scale Kronecker systems. The main feature of `Kronecker.jl` is providing a function `kronecker(A, B)` used to obtain an instance of the lazy `GeneralizedKroneckerProduct` type. In contrast to the native Julia function `kron(A, B)`, this does not compute the Kronecker product but instead stores the matrices in a specialized structure.

Commonly-used mathematical functions are overloaded to provide the most efficient methods to work with Kronecker products. We also provide an equivalent binary operator `⊗` which can be used directly as a Kronecker product in statements, i.e., `A ⊗ B`.

```@contents
Pages = [
    "man/basic.md",
    "man/types.md",
    "man/multiplication.md"
    "man/factorization.md",
    "man/indexed.md",
    "man/kroneckersums.md",
    "man/kroneckerpowers.md"
]
```

## Package features

- `tr`, `det`, `size`, `eltype`, `inv`, ... are efficient functions to work with Kronecker products. Either the result is a numeric value, or returns a new `KroneckerProduct` type.
- Kronecker product - vector multiplications are performed using the vec trick. Two Kronecker products of conformable size can be multiplied efficiently, yielding another Kronecker product.
- Working with incomplete systems using the [sampled vec trick](https://arxiv.org/pdf/1601.01507.pdf).
- Overloading of the function `eigen` to compute eigenvalue decompositions of Kronecker products. It can be used to efficiently solve systems of the form `(A ⊗ B +λI) \ v`.
- Higher-order Kronecker systems are supported: most functions work on `A ⊗ B ⊗ C` or systems of arbitrary order.
  - Efficient sampling of [Kronecker graphs](https://cs.stanford.edu/~jure/pubs/kronecker-jmlr10.pdf) is supported.
- Kronecker powers are supported: `kronecker(A, 3)` or `A ⊗ 3`.
- A `KroneckerSum` can be constructed with `A ⊕ B` (typed using `\oplus TAB`) or `kroneckersum(A,B)`.
  - Multiplication with vectors uses  a specialized version of the vec trick
  - Higher-order sums are supported, e.g. `A ⊕ B ⊕ C` or `kroneckersum(A,4)`.

## Example use

```@repl
using Kronecker, LinearAlgebra

A = [1.0 2.0;
     3.0 5.0];
B = Array{Float64, 2}([1 2 3;
            4 5 6;
            7 -2 9]);

K = A ⊗ B

collect(K)  # yield the dense matrix

tr(K)

det(K)

K'  # (conjugate transpose)

inv(K')

K * K  # (A * A) ⊗ (B * B)

v = collect(1:6)

K * v
```
