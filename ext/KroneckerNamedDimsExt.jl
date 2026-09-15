module KroneckerNamedDimsExt

using Kronecker, NamedDims
using Kronecker: KroneckerProduct, IndexedKroneckerProduct, KroneckerDiagonal,
    kron_names
import Kronecker: kronecker

# Re-wrap with names:
kronecker(A::NamedDimsArray{L1}, B::NamedDimsArray{L2}) where {L1,L2} =
    NamedDimsArray(KroneckerProduct(parent(A), parent(B)), kron_names(L1, L2))

kronecker(A::NamedDimsArray{L}, B::AbstractMatrix) where {L} =
    NamedDimsArray(KroneckerProduct(parent(A), B), kron_names(L, (:_, :_)))
kronecker(A::AbstractMatrix, B::NamedDimsArray{L}) where {L} =
    NamedDimsArray(KroneckerProduct(A, parent(B)), kron_names((:_, :_), L))

# Power
kronecker(A::NamedDimsArray{L}, p::Int) where {L} =
    NamedDimsArray(kronecker(parent(A), p), kron_names(L, Val(p)))

# Disambiguation between NamedDims' `*` methods (which dispatch on
# NamedDimsArray operands) and the Kronecker vec-trick methods. The type
# patterns mirror the signatures in NamedDims/src/functions_math.jl.
const NamedVector = NamedDimsArray{L,T,1,A} where {L,T,A<:AbstractVector{T}}

Base.:*(K::GeneralizedKroneckerProduct, v::NamedVector) = K * parent(v)
Base.:*(K::IndexedKroneckerProduct, v::NamedVector) = K * parent(v)
Base.:*(K::KroneckerDiagonal, v::NamedVector) = K * parent(v)

Base.:*(K::GeneralizedKroneckerProduct, M::NamedDimsArray{L,T,2,A}) where {L,T,A<:AbstractMatrix{T}} =
    NamedDimsArray(K * parent(M), (:_, last(L)))
Base.:*(M::NamedDimsArray{L,T,2,A}, K::GeneralizedKroneckerProduct) where {L,T,A<:AbstractMatrix{T}} =
    NamedDimsArray(parent(M) * K, (first(L), :_))

end # module
