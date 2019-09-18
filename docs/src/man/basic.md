# Basic use

Compute a lazy Kronecker products between two matrices `A` and `B` by either

```julia
K = kronecker(A, B)
```

or, by using the binary operator:

```julia
K = A ⊗ B
```

Note, `⊗` can be formed by typing `\otimes<tab>`.

The Kronecker product `K` behaves like a matrix, for which `size(K)`, `eltype(K)` works as one would expect. Elements can be accessed via `K[i,j]`; every element is computed on the fly. The function `collect` can be used to turn `K` in a regular, dense matrix.

```@repl
using Kronecker
A = randn(4, 4)
B = rand(1:10, 5, 7)
K = A ⊗ B
K[4, 5]
eltype(K)  # promotion
collect(K)
```

## Constructing Kronecker products

```@docs
kronecker
⊗
collect(::AbstractKroneckerProduct)
```

## Basic properties of Kronecker products

```@docs
getindex
eltype
size(::AbstractKroneckerProduct)
order
getmatrices
issquare
issymmetric
sum
```

## Linear algebra

Many functions of the `LinearAlgebra` module are overloaded to work with subtypes of `GeneralizedKroneckerProduct`.

```@docs
det(K::AbstractKroneckerProduct)
logdet(K::AbstractKroneckerProduct)
tr(K::AbstractKroneckerProduct)
inv(K::AbstractKroneckerProduct)
adjoint(K::AbstractKroneckerProduct)
transpose(K::AbstractKroneckerProduct)
conj(K::AbstractKroneckerProduct)
```
