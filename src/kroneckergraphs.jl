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
using Random: AbstractRNG, default_rng

"""
    isprob(A::AbstractArray)

Test if a matrix can be interpeted as a probability matrix, i.e., all elements
are between 0 and 1.
"""
isprob(A::AbstractArray) = all(0 .<= A .<= 1)

"""
    isprob(K::AbstractKroneckerProduct)

Test if a Kronecker product can be interpeted as a probability matrix,
i.e., all elements are between 0 and 1.
"""
function isprob(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return isprob(A) && isprob(B)
end

"""
    _sample_weighted(rng::AbstractRNG, items, weights, s::Int)

Draw `s` elements from `items` with replacement, with probabilities
proportional to `weights` (assumed non-negative).
"""
function _sample_weighted(rng::AbstractRNG, items, weights, s::Int)
    cw = cumsum(weights)
    total = last(cw)
    total > 0 || throw(ArgumentError("weights must have a positive sum"))
    return [items[searchsortedfirst(cw, rand(rng) * total)] for _ in 1:s]
end

"""
    naivesample([rng::AbstractRNG,] P::AbstractKroneckerProduct)

Sample a Kronecker graph from a probabilistic Kronecker product P using the
naive method. This method has a time complexity in the size of the Kronecker
product (but is still light in memory use). Consider using `fastsample`.
"""
function naivesample(rng::AbstractRNG, P::AbstractKroneckerProduct)
    @assert isprob(P) throw(DomainError(
        "All values of K should be between 0 and 1"))
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
    A, B = getmatrices(K)
    p, q = size(B)
    indicesA = sampleindices(rng, A, s)
    indicesB = sampleindices(rng, B, s)
    indices = similar(indicesA)
    for (o, (Ia, Ib)) in enumerate(zip(indicesA, indicesB))
        (i, j), (k, l) = Ia, Ib
        @inbounds indices[o] = ((i - 1) * p + k, (j - 1) * q + l)
    end
    return indices
end

sampleindices(A::AbstractMatrix, s::Int) = sampleindices(default_rng(), A, s)

"""
    fastsample([rng::AbstractRNG,] P::AbstractKroneckerProduct)

Uses the heuristic sampling from Leskovec et al. (2008) to sample a large
Kronecker graph.
"""
function fastsample(rng::AbstractRNG, P::AbstractKroneckerProduct)
    @assert isprob(P) throw(DomainError(
        "All values of K should be between 0 and 1"))
    G = spzeros(Bool, size(P)...)
    n = Int(round(sum(P)))  # expected number of edges
    for (i, j) in sampleindices(rng, P, n)
        @inbounds G[i, j] = true
    end
    return G
end

fastsample(P::AbstractKroneckerProduct) = fastsample(default_rng(), P)
