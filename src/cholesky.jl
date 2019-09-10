import Base: getproperty
import LinearAlgebra: cholesky, Cholesky, char_uplo, UpperTriangular, LowerTriangular, \,
    istril, istriu

const KroneckerCholesky{T} = Cholesky{T, <:AbstractKroneckerProduct{T}} where {T}

"""
    cholesky(K::AbstractKroneckerProduct; check=true)

Compute the Cholesky factorization of a product `K` of symmetric and positive
definite matrices and return a Cholesky factorization, with the Kronecker
structure retained. The following functions are available for Cholesky
factorizations of Kronecker products: `size`, `\`, `inv`, `det`, `logdet` and
`isposdef`.

See documentation of `LinearAlgebra.cholesky` for details.
"""
function cholesky(K::AbstractKroneckerProduct; check=true)
    A, B = getmatrices(K)
    chol_A, chol_B = cholesky(A; check=true), cholesky(B; check=true)
    return Cholesky(chol_A.factors ⊗ chol_B.factors, 'U', 0)
end

function Cholesky(factors::KroneckerProduct{T}, uplo::AbstractChar, info::Integer) where {T}
    return Cholesky{T, typeof(factors)}(factors, uplo, info)
end

"""
    UpperTriangular(K::AbstractKroneckerProduct)

Converts a `AbstractKroneckerProduct` by taking the upper triangular part of
the individual matrices.

Generally NOT the same matrix as `UpperTriangular(collect(K))`.
"""
function UpperTriangular(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return UpperTriangular(A) ⊗ UpperTriangular(B)
end

"""
    LowerTriangular(K::AbstractKroneckerProduct)

Converts a `AbstractKroneckerProduct` by taking the lower triangular part of
the individual matrices.

Generally NOT the same matrix as `LowerTriangular(collect(K))`.
"""
function LowerTriangular(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return LowerTriangular(A) ⊗ LowerTriangular(B)
end

"""
    istril(K::AbstractKroneckerProduct)

Checks if all matrices in an `AbstractKroneckerProduct` are lower triangular.
Implies that `K` is lower triangular.
"""
function istril(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return istril(A) && istril(B)
end

"""
    istriu(K::AbstractKroneckerProduct)

Checks if all matrices in an `AbstractKroneckerProduct` are upper triangular.
Implies that `K` is upper triangular.
"""
function istriu(C::AbstractKroneckerProduct)
    A, B = getmatrices(C)
    return istriu(A) && istriu(B)
end

function getproperty(C::KroneckerCholesky, d::Symbol)
    Cfactors = getfield(C, :factors)
    Cuplo    = getfield(C, :uplo)
    info     = getfield(C, :info)

    Cuplo != 'U' && throw(NotImplementedError(""))

    if d == :U
        return UpperTriangular(Cfactors)
    elseif d == :L
        return LowerTriangular(copy(Cfactors'))
        #return LowerTriangular(Cuplo === char_uplo(d) ? Cfactors : copy(Cfactors'))
    elseif d == :UL
        return (Cuplo === 'U' ? UpperTriangular(Cfactors) : LowerTriangular(Cfactors))
    else
        return getfield(C, d)
    end
end

"""
    logdet(K::KroneckerCholesky)

Compute the logarithm of the determinant of the Cholesky factorization of a
Kronecker product.
"""
function logdet(C::KroneckerCholesky)
    A, B = getmatrices(C.factors)
    logdet_A = logdet(Cholesky(A, C.uplo, 0))
    logdet_B = logdet(Cholesky(B, C.uplo, 0))
    return size(B, 1) * logdet_A + size(A, 1) * logdet_B
end

"""
    inv(C::KroneckerCholesky)

Compute the inverse of the Cholesky factorization of a Kronecker product.
"""
function Base.inv(C::KroneckerCholesky)
    A, B = getmatrices(C.factors)
    invA = inv(Cholesky(A, C.uplo, 0))
    invB = inv(Cholesky(B, C.uplo, 0))
    return invA ⊗ invB
end

function \(C::KroneckerCholesky, v::AbstractVecOrMat)
    return inv(C) * v
end
