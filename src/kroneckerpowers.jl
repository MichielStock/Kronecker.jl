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
        return new{typeof(A), pow}(A, pow)
    end
end

#TODO getindex
#TODO size
#TODO overloading all function
