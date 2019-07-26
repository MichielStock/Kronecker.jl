#=
Created on Friday 26 July 2019
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Standard matrix factorization algorithms applied on Kronecker systems.
=#


abstract type FactorizedKronecker <: AbstractSquareKronecker end

# CHOLESKY DECOMPOSITION
# ----------------------

import LinearAlgebra: Cholesky, cholesky

struct CholeskyKronecker{T<:Union{Cholesky,FactorizedKronecker},S<:Union{Cholesky,FactorizedKronecker}} <: FactorizedKronecker
    A::T
    B::S
end

_getuls(C::Cholesky) = C.UL
_getuls(C::CholeskyKronecker) = (_getuls(C.A), _getuls(C.B))

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, C::CholeskyKronecker)
    summary(io, C); println(io)
    println(io, "U factor:")
    show(io, mime, kronecker(_getuls(C)...))
end

# TODO: complete docstring!
function cholesky(K::SquareKroneckerProduct; check = true)
    A, B = getmatrices(K)
    return CholeskyKronecker(cholesky(A, check=check),
                            cholesky(B, check=check))
end
