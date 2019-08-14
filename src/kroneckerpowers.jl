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

"""
    kronecker(A::AbstractMatrix, pow::Int)

Kronecker power, computes `A ⊗ A ⊗ ... ⊗ A`. Returns a lazy `KroneckerPower`
type.
"""
kronecker(A::AbstractMatrix, pow::Int) = KroneckerPower(A, pow)

⊗(A::AbstractMatrix, pow::Int) = kronecker(A, pow)

getmatrices(K::KroneckerPower{T, N}) where {T, N} = (K.A, KroneckerPower(K.A, K.pow-1))
getmatrices(K::KroneckerPower{T, 2}) where {T} = (K.A, K.A)
getmatrices(K::KroneckerPower{T, 1}) where {T} = (K.A, )

order(K::KroneckerPower) = K.pow
Base.size(K::KroneckerPower) = size(K.A).^K.pow
Base.eltype(K::KroneckerPower) = eltype(K.A)
issquare(K::KroneckerPower) = issquare(K.A)

# SCALAR EQUIVALENTS FOR AbstractKroneckerProduct

"""
    det(K::KroneckerPower)

Compute the determinant of a Kronecker power.
"""
function LinearAlgebra.det(K::KroneckerPower)
    squarecheck(K)
    A, pow = K.A, K.pow
    n = size(A, 1)
    return det(K.A)^(n * pow)
end

"""
    logdet(K::KroneckerPower)

Compute the logarithm of the determinant of a Kronecker power.
"""
function LinearAlgebra.logdet(K::KroneckerPower)
    squarecheck(K)
    A, pow = K.A, K.pow
    n = size(A, 1)
    return n * pow * logdet(K.A)
end

"""
    tr(K::KroneckerPower)

Compute the trace of a Kronecker power.
"""
function LinearAlgebra.tr(K::KroneckerPower)
    squarecheck(K)
    return tr(K.A)^K.pow
end

# Matrix operations

"""
    inv(K::KroneckerPower)

Compute the inverse of a Kronecker power.
"""
function Base.inv(K::KroneckerPower)
    squarecheck(K)
    return KroneckerPower(inv(K.A), K.pow)
end

"""
    adjoint(K::KroneckerPower)

Compute the adjoint of a Kronecker power.
"""
function Base.adjoint(K::KroneckerPower)
    return KroneckerPower(K.A', K.pow)
end

"""
    transpose(K::KroneckerPower)

Compute the transpose of a Kronecker power.
"""
function Base.transpose(K::KroneckerPower)
    A, B = getmatrices(K)
    return KroneckerPower(transpose(K.A), K.pow)
end

function Base.conj(K::KroneckerPower)
    return KroneckerPower(conj(K.A), K.pow)
end

# mixed-product property
function Base.:*(K1::KroneckerPower{T1, N},
                        K2::KroneckerPower{T2, N}) where {T1, T2, N}
    if size(K1, 2) != size(K2, 1)
        throw(DimensionMismatch("Mismatch between K1 and K2"))
    end
    return KroneckerPower(K1.A * K2.A, N)
end
