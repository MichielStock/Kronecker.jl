# Types

The abstract type at the top of the hierarchy of `Kronecker.jl`'s type system is `GeneralizedKroneckerProduct` a subtype `AbstractMatrix`. `GeneralizedKroneckerProduct` contains all subtypes which contain a Kronecker product.

Pure Kronecker products, i.e., all expressions that one can write as `A ⊗ B`, with `A` and `B` `AbstractMatix` types are part of the abstract type `AbstractKroneckerProduct <: GeneralizedKroneckerProduct`. Concrete instantiations are stored in the structure `KroneckerProduct <: AbstractKroneckerProduct`, a container for `A` and `B`. Instances of `KroneckerProduct` structs are annotated with the element type of the Kronecker product (promoted from the element types of `A` and `B`) and the types of `A` and `B`.

For Kronecker powers, iterative multiplications of the same matrix, i.e.,

```math
\bigotimes_{i=1}^K A = A\otimes A \otimes \ldots \otimes A\,,
```

are stored in the structure `KroneckerPower <: AbstractKroneckerProduct`. This is more efficient, as it only processes a single matrix, irregardless of the order of the product.

Special cases are `KroneckerSum <: AbstractKroneckerSum <: GeneralizedKroneckerProduct` for the Kronecker sum:

```math
A \oplus B = A \otimes I + I \otimes B\,.
```

These work similar to instances of `AbstractKroneckerProduct`.

Finally, we have `IndexedKroneckerProduct <: GeneralizedKroneckerProduct`, which stores submatrices of a Kronecker product. This contains both the Kronecker product as well as the indices.

It is important to note that since all instances of subtypes of `GeneralizedKroneckerProduct` are instances of an `AbstractMatrix`, it is possible to combine them at heart. This is because Kronecker products are between any types of matrices, which Kronecker products themselves are.
