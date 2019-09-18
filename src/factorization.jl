#=
Created on Friday 26 July 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Standard matrix factorization algorithms applied on Kronecker systems.
=#


abstract type FactorizedKronecker <: AbstractKroneckerProduct end

# CHOLESKY DECOMPOSITION
# ----------------------

import LinearAlgebra: Cholesky, cholesky

struct CholeskyKronecker{T<:Union{Cholesky,FactorizedKronecker},S<:Union{Cholesky,FactorizedKronecker}} <: FactorizedKronecker
    A::T
    B::S
end

issquare(C::Cholesky) = true

function Base.getproperty(C::CholeskyKronecker, d::Symbol)
    if d in [:U, :L, :UL]
        return kronecker(getproperty(C.A, d), getproperty(C.B, d))
    elseif d in [:A, :B]
        return getfield(C, d)
    else
        throw(ArgumentError("Attribute :$d not supported (only :A, :B, :UL, :U or :L)"))
    end
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, C::CholeskyKronecker)
    summary(io, C); println(io)
    println(io, "U factor:")
    show(io, mime, getproperty(C, :U))
end

"""
    cholesky(K::AbstractKroneckerProduct; check = true)

Wrapper around `cholesky` from the `LinearAlgebra` package. Performs Cholesky
on the matrices of a `AbstractKroneckerProduct` instances and returns a
`CholeskyKronecker` type. Similar to `Cholesky`, `size`, `\\`, `inv`, `det`,
and `logdet` are overloaded to efficiently work with this type.
"""
function cholesky(K::AbstractKroneckerProduct; check = true)
    squarecheck(K)
    A, B = getmatrices(K)
    return CholeskyKronecker(cholesky(A, check=check),
                            cholesky(B, check=check))
end
