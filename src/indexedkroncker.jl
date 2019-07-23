Index = Union{UnitRange{I}, AbstractVector{I}} where I <: Int #TODO: make for general indices

struct IndexedKroneckerProduct <: GeneralizedKroneckerProduct
    K::AbstractKroneckerProduct
    p::Index
    q::Index
    r::Index
    t::Index
    function IndexedKroneckerProduct(K, p::Index, q::Index, r::Index, t::Index)
        if order(K) != 2
            throw(DimensionMismatch(
                "Indexed Kronecker only implemented for second order Kronecker systems"
            ))
        elseif !(length(p) == length(q) && length(r) == length(t))
            throw(DimensionMismatch("Indices should have matching lengths"))
        elseif minimum((minimum.((p, q, r, t)))) <= 0
            throw(BoundsError("Negative indices not allowed"))
        end
        A, B = getmatrices(K)
        (m, n) = size(B)
        (k, l) = size(A)
        if !(maximum(p) ≤ m && maximum(q) ≤ k)
            throw(BoundsError("Indices exeed matrix bounds"))
        end
        if !(maximum(r) ≤ n && maximum(t) ≤ l)
            throw(BoundsError("Indices exeed matrix bounds"))
        end
        return new(K, p, q, r, t)
    end
end

# Best way to do multi-line strings?

_dm(str) = throw(DimensionMismatch(str))
_ibe(str) = throw(BoundsError(str))

getindices(K::IndexedKroneckerProduct) = K.p, K.q, K.r, K.t

getmatrices(K::IndexedKroneckerProduct) = getmatrices(K.K)

Base.size(K::IndexedKroneckerProduct) = (length(K.p), length(K.t))

function Base.getindex(K::IndexedKroneckerProduct, i::Int, j::Int)
    A, B = getmatrices(K)
    p, q, r, t = getindices(K)
    return B[p[i],r[j]] * A[q[i],t[j]]
end

function Base.getindex(K::AbstractKroneckerProduct, p::Index, q::Index, r::Index, t::Index)
    N, M = getmatrices(K)
    a, b = size(M)
    c, d = size(N)
    return IndexedKroneckerProduct(K, p, q, r, t)
end

Base.eltype(K::IndexedKroneckerProduct) = eltype(K.K)

function Base.collect(K::IndexedKroneckerProduct)
    f, e = size(K)
    Kexplicit = Array{eltype(K)}(undef, f, e)
    N, M = getmatrices(K.K)
    p = K.p
    q = K.q
    r = K.r
    t = K.t
    for h in 1:f, g in 1:e
        Kexplicit[h, g] = N[q[h], t[g]] * M[p[h], r[g]]
    end
    return Kexplicit
end

function genvectrick!(M, N, v, u, p, q, r, t)
    # computes N ⊗ M

    a, b = size(M)
    c, d = size(N)
    e = length(v)
    f = length(u)

    @assert maximum(p) ≤ a && maximum(q) ≤ c
    @assert maximum(r) ≤ b && maximum(t) ≤ d
    u .= 0  # reset for inplace
    if a * e + d * f < c * e + b *f
        # compute T = VM'
        T = zeros(eltype(v), d, a)
        @simd for h in 1:e
            i, j = r[h], t[h]
            @simd for k in 1:a
                @inbounds T[j,k] += v[h] * M[k,i]
            end
        end
        @simd for h in 1:f
            i, j = p[h], q[h]
            @simd for k in 1:d
                @inbounds u[h] += N[j,k] * T[k,i]
            end
        end
    else
        # compute S = NV
        S = zeros(eltype(v), d, a)
        @simd for h in 1:e
            i, j = r[h], t[h]
            @simd for k in 1:c
                @inbounds S[k,j] += v[h] * N[k,j]
            end
        end
        @simd for h in 1:f
            i, j = p[h], q[h]
            @simd for k in 1:b
                @inbounds u[h] += S[j,k] * M[i,k]
            end
        end
    end
    return u
end

function genvectrick2!(M, N, v, u, p, q, r, t)
    # computes N ⊗ M
    a, b = size(M)
    c, d = size(N)
    e = length(v)
    f = length(u)

    u .= 0  # reset for inplace
    if a * e + d * f < c * e + b *f
        # compute T = VM'
        T = zeros(eltype(v), d, a)
        @simd for k in 1:a
            @simd for h in 1:e
                i, j = r[h], t[h]
                @inbounds T[j,k] += v[h] * M[k,i]
            end
        end
        @simd for k in 1:d
            @simd for h in 1:f
                i, j = p[h], q[h]
                @inbounds u[h] += N[j,k] * T[k,i]
            end
        end
    else
        # compute S = NV
        S = zeros(eltype(v), d, a)

        @simd for k in 1:c
            @simd for h in 1:e
                i, j = r[h], t[h]
                @inbounds S[k,j] += v[h] * N[k,j]
            end
        end

        @simd for k in 1:b
            @simd for h in 1:f
                i, j = p[h], q[h]
                @inbounds u[h] += S[j,k] * M[i,k]
            end
        end
    end
    return u
end

function genvectrick(M, N, v, p, q, r, t)
    # computes N ⊗ M
    f = length(p)
    u = zeros(eltype(v), f)
    return genvectrick!(M, N, v, u, p, q, r, t)
end

function Base.:*(K::IndexedKroneckerProduct, v::AbstractVector)
    if length(v) != size(K, 2)
        throw(DimensionMismatch(string(
            "Size indexed Kronecker system ($(size(K))) does not ",
            "match vector size ($(size(v)))",
        )))
    end
    N, M = getmatrices(K.K)
    p, q, r, t = getindices(K)
    return genvectrick(M, N, v, p, q, r, t)
end
