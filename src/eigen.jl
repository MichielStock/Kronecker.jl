using LinearAlgebra: Eigen
import LinearAlgebra: eigen, \, det, logdet, inv
import Base: +

function eigen(K::AbstractKroneckerProduct)
    squarecheck(K)
    A_λ, A_Γ = eigen(K.A)
    B_λ, B_Γ = eigen(K.B)
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

function \(E::Eigen{<:Real, <:Real, <:AbstractKroneckerProduct}, v::AbstractVector{<:Real})
    λ, Γ = E
    return Γ * (Diagonal(λ) \ (Γ' * v))
end

det(E::Eigen{<:Real, <:Real, <:AbstractKroneckerProduct}) = prod(E.values)
logdet(E::Eigen{<:Real, <:Real, <:AbstractKroneckerProduct}) = sum(log, E.values)
inv(E::Eigen{<:Real, <:Real, <:AbstractKroneckerProduct}) = Eigen(inv.(E.values), E.vectors)
