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
struct KroneckerPower{T,TA<:AbstractMatrix{T}} <: AbstractKroneckerProduct{T}
    A::TA
    pow::Int
    function KroneckerPower(A::AbstractMatrix, pow::Integer)
        @assert pow ≥ 2 "KroneckerPower only makes sense for powers greater than 1"
        return new{eltype(A),typeof(A)}(A, Int(pow))
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

getallfactors(K::KroneckerPower) = ntuple(_ -> K.A, K.pow)

getmatrices(K::KroneckerPower) = (K.pow == 2 ? K.A : KroneckerPower(K.A, K.pow - 1), K.A)
lastmatrix(K::KroneckerPower) = K.A

order(K::KroneckerPower) = K.pow
Base.size(K::KroneckerPower) = size(K.A) .^ K.pow
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

function LinearAlgebra.diag(K::KroneckerPower)
    if issquare(K.A)
        d = reshape(diag(K.A), :, 1)
        return vec(reduce(kron, fill(d, order(K))))
    end
    return K[diagind(K)]
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
function _mulmixed(K1::KroneckerPower, K2::KroneckerPower)
    if size(K1, 2) != size(K2, 1)
        throw(DimensionMismatch("Mismatch between K1 and K2"))
    end
    return KroneckerPower(K1.A * K2.A, K1.pow)
end
function Base.:*(K1::KroneckerPower, K2::KroneckerPower)
    K1.pow == K2.pow || throw(ArgumentError("multiplication is only defined if all terms have the same exponent"))
    _mulmixed(K1, K2)
end
const KronPowDiagonal = KroneckerPower{<:Any,<:Diagonal}
function Base.:*(K1::KronPowDiagonal, K2::KronPowDiagonal)
    K1.pow == K2.pow || throw(ArgumentError("multiplication is only defined if all terms have the same exponent"))
    _mulmixed(K1, K2)
end

for T in [:Diagonal, :UniformScaling]
    @eval Base.:+(K::KronPowDiagonal, D::$T) = Diagonal(K) + D
    @eval Base.:+(D::$T, K::KronPowDiagonal) = D + Diagonal(K)
    @eval Base.:-(K::KronPowDiagonal, D::$T) = Diagonal(K) - D
    @eval Base.:-(D::$T, K::KronPowDiagonal) = D - Diagonal(K)
end

"""
    lmul!(a::Number, K::KroneckerPower)

Scale an `KroneckerPower` `K` inplace by a factor `a` by rescaling the matrix the base
matrix with a factor `a^(1/N)`.

It is recommended to rewrite your Kronecker product rather as `copy(A) ⊗ (A ⊗ n - 1)`
(note the copy) for numerical stability. This will only modify the first matrix, leaving
the chain of Kronecker products alone.
"""
function LinearAlgebra.lmul!(a::Number, K::KroneckerPower)
    # will only work if eltype is some kind of float
    #eltype(K) <: AbstractFloat || throw(InexactError)
    n = order(K)
    @warn "Inplace scaling of a Kronecker product, consider rewriting as `copy(A) ⊗ (A ⊗ $n - 1)`"
    A = K.A
    lmul!(a^(1 / n), A)
end
