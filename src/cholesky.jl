import Base: getproperty
import LinearAlgebra: cholesky, Cholesky, char_uplo, UpperTriangular, LowerTriangular, \,
    istril, istriu

const KroneckerCholesky{T} = Cholesky{T, <:AbstractKroneckerProduct{T}} where {T}

function cholesky(K::AbstractKroneckerProduct; check=true)
    A, B = getmatrices(K)
    chol_A, chol_B = cholesky(A; check=true), cholesky(B; check=true)
    return Cholesky(chol_A.factors ⊗ chol_B.factors, 'U', 0)
end

function Cholesky(factors::KroneckerProduct{T}, uplo::AbstractChar, info::Integer) where {T}
    return Cholesky{T, typeof(factors)}(factors, uplo, info)
end

function UpperTriangular(C::AbstractKroneckerProduct)
    A, B = getmatrices(C)
    return UpperTriangular(A) ⊗ UpperTriangular(B)
end

function LowerTriangular(C::AbstractKroneckerProduct)
    A, B = getmatrices(C)
    return LowerTriangular(A) ⊗ LowerTriangular(B)
end

function istril(C::AbstractKroneckerProduct)
    A, B = getmatrices(C)
    return istril(A) && istril(B)
end

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

function logdet(C::KroneckerCholesky)
    A, B = getmatrices(C.factors)
    logdet_A = logdet(Cholesky(A, C.uplo, 0))
    logdet_B = logdet(Cholesky(B, C.uplo, 0))
    return size(B, 1) * logdet_A + size(A, 1) * logdet_B
end

function Base.inv(C::KroneckerCholesky)
    A, B = getmatrices(C.factors)
    invA = inv(Cholesky(A, C.uplo, 0))
    invB = inv(Cholesky(B, C.uplo, 0))
    return invA ⊗ invB
end

function \(C::KroneckerCholesky, x::AbstractVecOrMat)
    return inv(C) * x
end
