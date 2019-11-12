using NamedDims

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

# Finding the names
kron_names(left::Tuple, right::Tuple) = map(_join, left, right)

_join(i::Symbol, j::Symbol) = _join(Val(i), Val(j))
@generated _join(::Val{i}, ::Val{j}) where {i,j} = QuoteNode(Symbol(i, :ᵡ, j))

kron_names(L::Tuple, ::Val{1}) = L
kron_names(L::Tuple, ::Val{p}) where {p} = kron_names(kron_names(L,L), Val(p-1))
