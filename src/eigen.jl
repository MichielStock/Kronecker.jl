using LinearAlgebra: Eigen
import LinearAlgebra: eigen, \, det, logdet, inv
import Base: +

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

det(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct}) = prod(E.values)
logdet(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct}) = sum(log, E.values)
inv(E::Eigen{<:Number, <:Number, <:AbstractKroneckerProduct}) = Eigen(inv.(E.values), E.vectors)
