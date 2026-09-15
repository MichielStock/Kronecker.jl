# Helpers to combine dimension names of Kronecker factors, e.g. :i and :j
# into :iᵡj. They act on plain symbols and tuples, so they live in the package
# proper; the methods that dispatch on NamedDimsArray are defined in the
# package extension `ext/KroneckerNamedDimsExt.jl`.

kron_names(left::Tuple, right::Tuple) = map(_join, left, right)

_join(i::Symbol, j::Symbol) = _join(Val(i), Val(j))
@generated _join(::Val{i}, ::Val{j}) where {i,j} = QuoteNode(Symbol(i, :ᵡ, j))

kron_names(L::Tuple, ::Val{1}) = L
kron_names(L::Tuple, ::Val{p}) where {p} = kron_names(kron_names(L, L), Val(p - 1))
