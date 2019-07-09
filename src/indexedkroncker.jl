#Index = Union{UnitRange{I}, Array{I,1}} where I <: Int
Index = Union{UnitRange{I}, AbstractVector{I}} where I <: Int #TODO: make for general indices

struct IndexedKroneckerProduct <: GeneralizedKroneckerProduct
    K::AbstractKroneckerProduct
    p::Index
    q::Index
    r::Index
    t::Index
    function IndexedKroneckerProduct(K, p, q, r, t)
        order(K) == 2 || throw(DimensionMismatch("Indexed Kronecker only implemented for second order Kronecker systems"))
        length(p) == length(q) && length(r) == length(t) || throw(DimensionMismatch("Indices should have matching lengths"))
        minimum((minimum.((p, q, r, t)))) > 0 || throw(BoundsError("Negative indices not allowed"))
        A, B = getmatrices(K)
        (m, n) = size(B)
        (k, l) = size(A)
        maximum(p) ≤ m && maximum(q) ≤ k || throw(BoundsError("Indices exeed matrix bounds"))
        maximum(r) ≤ n && maximum(t) ≤ l || throw(BoundsError("Indices exeed matrix bounds"))
        new(K, p, q, r, t)
    end
end

function getindices(K::IndexedKroneckerProduct)
    return K.p, K.q, K.r, K.t
end

getmatrices(K::IndexedKroneckerProduct) = getmatrices(K.K)

function Base.:size(K::IndexedKroneckerProduct)
    return (length(K.p), length(K.t))
end

function Base.:getindex(K::IndexedKroneckerProduct, i::Int, j::Int)
    A, B = getmatrices(K)
    p, q, r, t = getindices(K)
    return B[p[i],r[j]] * A[q[i],t[j]]
end

function Base.:getindex(K::T, p::Index, q::Index, r::Index, t::Index) where T <: AbstractKroneckerProduct
    N, M = getmatrices(K)
    a, b = size(M)
    c, d = size(N)
    return IndexedKroneckerProduct(K, p, q, r, t)
end

function Base.:eltype(K::T) where T <: IndexedKroneckerProduct
    return eltype(K.K)
end

function Base.:collect(K::T) where T <: IndexedKroneckerProduct
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
    # sizes
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
    # sizes
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

function Base.:*(K::IndexedKroneckerProduct, v::V where V<:AbstractVector)
    (length(v) == size(K, 2)) || throw(DimensionMismatch("Size indexed Kronecker system ($(size(K))) does not match vector size ($(size(v)))"))
    N, M = getmatrices(K.K)
    p, q, r, t = getindices(K)
    return genvectrick(M, N, v, p, q, r, t)
end
