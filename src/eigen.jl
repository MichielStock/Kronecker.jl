using LinearAlgebra: Eigen
import LinearAlgebra: eigen, \, det, logdet, inv
import Base: +

"""
    eigen(K::AbstractKroneckerProduct)

Wrapper around `eigen` from the `LinearAlgebra` package. Performs Eigenvalue decompositon
on the matrices of a `AbstractKroneckerProduct` instances and returns an
`Eigen` type. The functions `size`, `\\`, `inv`, `det`,
and `logdet` are overloaded to efficiently work with this type.
"""
function eigen(K::AbstractKroneckerProduct)
    squarecheck(K)
    A, B = getmatrices(K)
    A_λ, A_Γ = eigen(A)
    B_λ, B_Γ = eigen(B)
    return Eigen(kron(A_λ, B_λ), kronecker(A_Γ, B_Γ))
end

+(E::Eigen, B::UniformScaling) = Eigen(E.values .+ B.λ, E.vectors)
+(A::UniformScaling, E::Eigen) = E + A

"""
    function collect(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct})

Collects eigenvalue decomposition of a `AbstractKroneckerProduct` type into a
matrix.
"""
function collect(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct})
    λ, Γ = E
    return Γ * Diagonal(λ) * Γ'
end

function \(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct}, v::AbstractVector{<:Number})
    λ, Γ = E
    return Γ * (Diagonal(λ) \ (Γ' * v))
end

"""
    det(K::Eigen)

Compute the determinant of the eigenvalue decomp of aKronecker product.
"""
det(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct}) = prod(E.values)

"""
    logdet(K::Eigen)

Compute the logarithm of the determinant of the eigenvalue decomp of aKronecker
product.
"""
logdet(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct}) = sum(log, E.values)

"""
    inv(K::Eigen)

Compute the inverse of the eigenvalue decomp of aKronecker product. Returns
another type of `Eigen`.
"""
inv(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct}) = Eigen(inv.(E.values), E.vectors)
