#=
Created on Monday 12 August 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Efficient ways of storing Kronecker powers.
=#

"""
Efficient way of storing Kronecker powers, e.g.

K = A ⊗ A ⊗ ... ⊗ A.
"""
struct KroneckerPower{TA<:AbstractMatrix, N} <: AbstractKroneckerProduct
   A::TA
   pow::Integer
   function KroneckerPower(A::AbstractMatrix{T}, pow::Integer) where {T}
      @assert pow ≥ 2 "KroneckerPower only makes sense for powers greater than 1"
      return new{typeof(A), pow}(A, pow)
    end
end

getmatrices(K::KroneckerPower{T, N}) where {T, N} = (K.A, KroneckerPower(K.A, K.pow-1))
getmatrices(K::KroneckerPower{T, 2}) where {T} = (K.A, K.A)
getmatrices(K::KroneckerPower{T, 1}) where {T} = (K.A, )

order(K::KroneckerPower) = K.pow
Base.size(K::KroneckerPower) = size(K.A).^K.pow
