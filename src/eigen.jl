using LinearAlgebra: Eigen
import LinearAlgebra: eigen, \, det, logdet, inv
import Base: +

function eigen(A::SquareKroneckerProduct)
    A_λ, A_Γ = eigen(A.A)
    B_λ, B_Γ = eigen(A.B)
    return Eigen(kron(A_λ, B_λ), kronecker(A_Γ, B_Γ))
end

+(A::Eigen, B::UniformScaling) = Eigen(A.values .+ B.λ, A.vectors)
+(A::UniformScaling, B::Eigen) = B + A

function \(A::Eigen{<:Real, <:Real, <:SquareKroneckerProduct}, v::AbstractVector{<:Real})
    λ, Γ = A
    return Γ * (Diagonal(λ) \ (Γ' * v))
end

det(A::Eigen{<:Real, <:Real, <:SquareKroneckerProduct}) = prod(A.values)
logdet(A::Eigen{<:Real, <:Real, <:SquareKroneckerProduct}) = sum(log, A.values)
inv(A::Eigen{<:Real, <:Real, <:SquareKroneckerProduct}) = Eigen(inv.(A.values), A.vectors)
