#=
Created on Saturday 3 August 2019
Last update: Thursday 8 August 2019

@author: Michiel Stock
michielfmstock@gmail.com

Some methods to simulate Kronecker graphs.

Leskovec, J., Chakrabarti, D., Kleinberg, J., Faloutsos, C., & Ghahramani,
Z. (2008). Kronecker graphs: an approach to modeling networks.
Journal of Machine Learning Research, 11, 985–1042.
Retrieved from https://cs.stanford.edu/~jure/pubs/kronecker-jmlr10.pdf
=#

using SparseArrays: spzeros
using Random: AbstractRNG, default_rng, rand!

"""
    isprob(A::AbstractArray)

Test if a matrix can be interpeted as a probability matrix, i.e., all elements
are between 0 and 1.
"""
isprob(A::AbstractArray) = all(a->0 ≤ a ≤ 1, A)

"""
    isprob(K::AbstractKroneckerProduct)

Test if a Kronecker product can be interpeted as a probability matrix,
i.e., all elements are between 0 and 1.
"""
@inline function isprob(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return isprob(A) && isprob(B)
end

"""
    _sample_weighted(rng::AbstractRNG, items, weights, s::Int)

Draw `s` elements from `items` with replacement, with probabilities
proportional to `weights` (assumed non-negative).
"""
function _sample_weighted(rng::AbstractRNG, items, weights, s::Int)
    any(w -> w < 0, weights) && throw(ArgumentError("weights must be non-negative"))
    cw = cumsum(weights)
    total = last(cw)
    total > 0 || throw(ArgumentError("weights must have a positive sum"))
    return [items[searchsortedfirst(cw, u * total)] for u in rand(rng, s)]
end

"""
    _accumulate_indices!(rows, cols, cw, is, js, u, m, n)

Fold one Kronecker factor into the running global indices: draw an entry of
the `m × n` factor for each uniform variate in `u` (via its cumulative weights
`cw` over the vectorised factor, with `is`/`js` mapping linear factor indices
back to subscripts) and update `rows`/`cols` Horner-style, i.e.
`i ← (i - 1)m + i_factor`.
"""
function _accumulate_indices!(rows::Vector{Int}, cols::Vector{Int},
    cw::AbstractVector, is::Vector{Int}, js::Vector{Int},
    u::Vector{Float64}, m::Int, n::Int)
    total = last(cw)
    @inbounds for o in eachindex(rows, cols, u)
        idx = _findfirstweight(cw, u[o] * total)
        rows[o] = (rows[o] - 1) * m + is[idx]
        cols[o] = (cols[o] - 1) * n + js[idx]
    end
    return nothing
end

# For the small factors typical of Kronecker graph seeds, a linear scan beats
# the branchy generic binary search; fall back to the latter for large factors.
@inline function _findfirstweight(cw::AbstractVector, t)
    length(cw) > 16 && return searchsortedfirst(cw, t)
    idx = 1
    @inbounds while cw[idx] < t
        idx += 1
    end
    return idx
end

"""
    naivesample([rng::AbstractRNG,] P::AbstractKroneckerProduct)

Sample a Kronecker graph from a probabilistic Kronecker product P using the
naive method. This method has a time complexity in the size of the Kronecker
product (but is still light in memory use). Consider using `fastsample`.
"""
function naivesample(rng::AbstractRNG, P::AbstractKroneckerProduct)
    isprob(P) || throw(DomainError(P,
        "All values of P should be between 0 and 1"))
    G = spzeros(Bool, size(P)...)
    for I in CartesianIndices(P)
        if P[I] > rand(rng)  # QUESTION: is this the most efficient way?
            @inbounds G[I] = true
        end
    end
    return G
end

naivesample(P::AbstractKroneckerProduct) = naivesample(default_rng(), P)

"""
    sampleindices([rng::AbstractRNG,] A::AbstractMatrix, s::Int)

Samples the indices from an `AbstractMatrix`. Probability of sampling indices is
proportional to the size of the corresponding value. Does not do any checks on A.
"""
sampleindices(rng::AbstractRNG, A::AbstractMatrix, s::Int) =
    Tuple.(_sample_weighted(rng, CartesianIndices(A), vec(A), s))

"""
    sampleindices([rng::AbstractRNG,] K::AbstractKroneckerProduct, s::Int)

Samples the indices from an `AbstractKroneckerProduct`. Probability of
sampling indices is proportional to the size of the corresponding value.
Does not do any checks on A.
"""
function sampleindices(rng::AbstractRNG, K::AbstractKroneckerProduct, s::Int)
    rows = ones(Int, s)
    cols = ones(Int, s)
    u = Vector{Float64}(undef, s)
    prev, cw, is, js = nothing, nothing, nothing, nothing
    for A in getallfactors(K)
        m = size(A, 1)
        if A !== prev  # a KroneckerPower repeats one factor: reuse its weights
            any(w -> w < 0, A) && throw(ArgumentError("factors must be non-negative"))
            cw = cumsum(vec(A))
            last(cw) > 0 || throw(ArgumentError("factors must have a positive sum"))
            is = [mod1(idx, m) for idx in eachindex(cw)]
            js = [fld1(idx, m) for idx in eachindex(cw)]
            prev = A
        end
        rand!(rng, u)
        _accumulate_indices!(rows, cols, cw, is, js, u, size(A)...)
    end
    return collect(zip(rows, cols))
end

sampleindices(A::AbstractMatrix, s::Int) = sampleindices(default_rng(), A, s)

"""
    fastsample([rng::AbstractRNG,] P::AbstractKroneckerProduct)

Uses the heuristic sampling from Leskovec et al. (2008) to sample a large
Kronecker graph: edges are drawn proportionally to their probability until the
expected number of edges `round(Int, sum(P))` is reached, re-sampling any
duplicates (collisions) along the way.
"""
function fastsample(rng::AbstractRNG, P::AbstractKroneckerProduct)
    isprob(P) || throw(DomainError(P,
        "All values of P should be between 0 and 1"))
    G = spzeros(Bool, size(P)...)
    n = round(Int, sum(P))  # expected number of edges
    added = 0
    while added < n
        for (i, j) in sampleindices(rng, P, n - added)
            if !G[i, j]
                G[i, j] = true
                added += 1
            end
        end
    end
    return G
end

fastsample(P::AbstractKroneckerProduct) = fastsample(default_rng(), P)
