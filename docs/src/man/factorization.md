# Factorization methods

For many forms of matrix factorization (eigenvalue decompositon, LU factorization, Cholesky factoization...), the decompositon of the Kronecker product is the Kronecker product of the decompostions. We have overloaded some of the factorization functions from `LinearAlgebra` to compute the factorization of instances of `AbstractKroneckerProduct`.

## Eigenvalue decompositon

The function `eigen` of `LinearAlgebra` is overloaded to compute the decompositon of `AbstractKroneckerProduct`s. The result is a factorization of the `Eigen` type, containing a vector of the eigenvalues and a matrix with the eigenvectors, just like long-time users would expect! The eigenvectors are structured as Kronecker products and can be processed accordingly.

The functions `det`, `logdet`, `inv` and `\` are overloaded the make use of this decompositon.

The eigenvalue decompositon of matrices can be used to solve large systems of the form:

```math
(A \otimes B + c\cdot I) \mathbf{x} = \mathbf{b}
```

The case where $A$ and $B$ are postive semi-definite occurs frequently in machine learning, for example in ridge regression.

```@repl
using Kronecker, LinearAlgebra # hide
A, B = rand(10, 10), randn(4, 4);
As, Bs = (A, B) .|> X -> X * X';  # make postive definite
K = As ⊗ Bs
E = eigen(K)
logdet(E)
b = randn(40);
(E + 0.1I) \ b  # solve a system
```

```@docs
eigen
+(E::Eigen, B::UniformScaling)
+(::Eigen, ::UniformScaling)

det(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct})
logdet(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct})
inv(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct})
```
