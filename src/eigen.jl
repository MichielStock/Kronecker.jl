using LinearAlgebra: Eigen
import LinearAlgebra: eigen, \, det, logdet, inv
import Base: +

function eigen(K::SquareKroneckerProduct)
    A_λ, A_Γ = eigen(K.A)
    B_λ, B_Γ = eigen(K.B)
    return Eigen(kron(A_λ, B_λ), kronecker(A_Γ, B_Γ))
end

+(E::Eigen, B::UniformScaling) = Eigen(E.values .+ B.λ, E.vectors)
+(A::UniformScaling, E::Eigen) = E + A

"""
    function collect(E::Eigen{<:Number, <:Number, <:SquareKroneckerProduct})

Collects eigenvalue decomposition of a `AbstractKroneckerProduct` type into a
matrix.
"""
function collect(E::Eigen{<:Number, <:Number, <:SquareKroneckerProduct})
    λ, Γ = E
    return Γ * Diagonal(λ) * Γ'
end

function \(E::Eigen{<:Real, <:Real, <:SquareKroneckerProduct}, v::AbstractVector{<:Real})
    λ, Γ = E
    return Γ * (Diagonal(λ) \ (Γ' * v))
end

det(E::Eigen{<:Real, <:Real, <:SquareKroneckerProduct}) = prod(E.values)
logdet(E::Eigen{<:Real, <:Real, <:SquareKroneckerProduct}) = sum(log, E.values)
inv(E::Eigen{<:Real, <:Real, <:SquareKroneckerProduct}) = Eigen(inv.(E.values), E.vectors)
