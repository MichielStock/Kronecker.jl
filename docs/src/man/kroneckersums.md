# Kronecker sums

A Kronecker sum between two square matrices of the same size is defined as

```math
A \oplus B = A ⊗ I + I \oplus B\,.
```

To construct objects of the `KroneckerSum` type, one can either use `kroneckersum` or the binary operator `⊕`. Lazy Kronecker sums work like lazy Kronecker products, though there are far fewer methods to process these constructs efficiently. The most important property for Kronecker sums relates to matrix exponentiation:
```math
\exp(A \oplus B) = \exp(A) \otimes \exp(B)\,.
```
The function `collect` can be used to transform a `KroneckerSum` struct into a sparse array. It is recommended to make it 'dense' this way before doing operations such as multiplication with a vector.

```@repl
using Kronecker # hide
A, B = rand(Bool, 5, 5), rand(4, 4);
K = A ⊕ B
exp(K)
collect(K)
```

```@docs
kroneckersum
⊕
collect(K::AbstractKroneckerSum)
exp
mul!(x::AbstractVector, K::AbstractKroneckerSum, v::AbstractVector)
```
