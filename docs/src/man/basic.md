# Basic use

Kronecker products between two matrices `A` and `B` by either

```julia
K = kronecker(A, B)
```

or using the binary operator:

```julia
K = A ⊗ B
```

The Kronecker product `K` behaves as a matrix, for which `size(K)`, `eltype(K)` works as expected. Elements can be accessed via `K[i,j]`, every element is computed on the fly. The function `collect` can be used to turn `K` in a regular, dense matrix.

## Constructing Kronecker products

```@docs
kronecker
\otimes
collect(K::AbstractKroneckerProduct)
```

## Basic properties of Kronecker products

```@docs
getindex(K::AbstractKroneckerProduct, i1::Int, i2::Int)
eltype(K::AbstractKroneckerProduct)
order
getmatrices
issquare
issymmetric
```
