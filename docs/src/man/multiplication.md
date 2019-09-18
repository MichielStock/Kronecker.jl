# Multiplication

`Kronecker.jl` allows for efficient multiplication of large Kronecker systems by overloading the multiplication function `*`. We distinguish three cases:

- **Kronecker-Kronecker multiplications** yield again a type of `AbstractKroneckerProduct`;
- **Kronecker-vector multiplications** use the 'vec trick' and yield a vector;
- **sampled Kronecker-vector multiplications** use the sampled-vec trick to yield a vector.

## Kronecker-kronecker multiplications

Multiplying two conformable Kronecker products of the same order yield a new Kronecker product, based on the mixed-product property:

```math
(A \otimes B)(C \otimes D) = (AC) \otimes (BD),
```

```@example
using Kronecker # hide
A, B, C, D = randn(5, 5), randn(4, 4), randn(5, 4), randn(4, 4);
(A ⊗ B) * (C ⊗ D)
```

## The Vec trick

Reshaping allows computing a product between a Kronecker product and vector as two matrix multiplications. This is the so-called vec trick which holds for any set of conformable matrices:

```math
(A \otimes B) \text{vec}(X) = \text{vec}(B^\intercal X A).
```

Here, $\text{vec}(\cdot)$ is the vectorization operator, which stacks all columns of a matrix into a vector.

```@example
using Kronecker # hide
A, B = rand(10, 10), rand(5, 6);
x = randn(60);
(A ⊗ B) * x
```

Note that this trick is extended to also work with matrices:

```@example
using Kronecker # hide
A, B = rand(10, 10), rand(5, 6);
x = randn(60, 2);
(A ⊗ B) * x
```

The vec trick works with higher-order Kronecker products. **However, at the moment this has a substantial overhead and likely be relatively slow.**

## Docstrings

```@docs
mul!
lmul!
rmul!
```

## Sampled Kronecker-vector multiplications

See [Indexed Kronecker products](@ref) for the specifics.
