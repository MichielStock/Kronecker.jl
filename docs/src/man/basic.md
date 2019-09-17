# Basic use

Compute a lazyronecker products between two matrices `A` and `B` by either

```julia
K = kronecker(A, B)
```

or using the binary operator:

```julia
K = A ⊗ B
```

Note, `⊗` can be formed by typing `\otimes<tab>`.

The Kronecker product `K` behaves as a matrix, for which `size(K)`, `eltype(K)` works as one would expect. Elements can be accessed via `K[i,j]`, every element is computed on the fly. The function `collect` can be used to turn `K` in a regular, dense matrix.

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
