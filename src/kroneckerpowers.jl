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
struct KroneckerPower{T<:Any,TA<:AbstractMatrix{T}, N} <: AbstractKroneckerProduct{T}
   A::TA
   pow::Integer
   function KroneckerPower(A::AbstractMatrix{T}, pow::Integer) where {T}
      @assert pow ≥ 2 "KroneckerPower only makes sense for powers greater than 1"
      return new{eltype(A), typeof(A), pow}(A, pow)
    end
end

"""
    kronecker(A::AbstractMatrix, pow::Int)

Kronecker power, computes `A ⊗ A ⊗ ... ⊗ A`. Returns a lazy `KroneckerPower`
type.
"""
kronecker(A::AbstractMatrix, pow::Integer) = KroneckerPower(A, pow)

"""
    ⊗(A::AbstractMatrix, pow::Int)

Kronecker power, computes `A ⊗ A ⊗ ... ⊗ A`. Returns a lazy `KroneckerPower`
type.
"""
⊗(A::AbstractMatrix, pow::Integer) = kronecker(A, pow)

getallfactors(K::KroneckerPower{T,TA,N}) where {T,TA,N} = ntuple(_ -> K.A, K.pow)

getmatrices(K::KroneckerPower{T,TA,N}) where {T,TA,N} = (K.A, KroneckerPower(K.A, K.pow-1))
getmatrices(K::KroneckerPower{T,TA,2}) where {T,TA} = (K.A, K.A)
getmatrices(K::KroneckerPower{T,TA,1}) where {T,TA} = (K.A, )

order(K::KroneckerPower) = K.pow
Base.size(K::KroneckerPower) = size(K.A).^K.pow
Base.eltype(K::KroneckerPower{T,TA,N}) where {T,TA,N} = T
issquare(K::KroneckerPower) = issquare(K.A)

# SCALAR EQUIVALENTS FOR AbstractKroneckerProduct

"""
    det(K::KroneckerPower)

Compute the determinant of a Kronecker power.
"""
function LinearAlgebra.det(K::KroneckerPower)
    checksquare(K.A)
    A, pow = K.A, K.pow
    n = size(A, 1)
    return det(K.A)^(n * pow)
end

"""
    logdet(K::KroneckerPower)

Compute the logarithm of the determinant of a Kronecker power.
"""
function LinearAlgebra.logdet(K::KroneckerPower)
    checksquare(K.A)
    A, pow = K.A, K.pow
    n = size(A, 1)
    return n * pow * logdet(K.A)
end

"""
    tr(K::KroneckerPower)

Compute the trace of a Kronecker power.
"""
function LinearAlgebra.tr(K::KroneckerPower)
    checksquare(K.A)
    return tr(K.A)^K.pow
end

# Matrix operations

"""
    inv(K::KroneckerPower)

Compute the inverse of a Kronecker power.
"""
function Base.inv(K::KroneckerPower)
    checksquare(K.A)
    return KroneckerPower(inv(K.A), K.pow)
end


"""
    pinv(K::KroneckerPower)

Compute the Moore-Penrose pseudo-inverse of a Kronecker power.
"""
function LinearAlgebra.pinv(K::KroneckerPower)
    return KroneckerPower(pinv(K.A), K.pow)
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
    return KroneckerPower(transpose(K.A), K.pow)
end

"""
    conj(K::KroneckerPower)

Compute the conjugate of a Kronecker power.
"""
function Base.conj(K::KroneckerPower)
    return KroneckerPower(conj(K.A), K.pow)
end

# mixed-product property
function Base.:*(K1::KroneckerPower{T,TA,N},
                        K2::KroneckerPower{S,TB,N}) where {T,TA,S,TB,N}
    if size(K1, 2) != size(K2, 1)
        throw(DimensionMismatch("Mismatch between K1 and K2"))
    end
    return KroneckerPower(K1.A * K2.A, N)
end
