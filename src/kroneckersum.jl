abstract type AbstractKroneckerSum <: GeneralizedKroneckerProduct end

struct KroneckerSum{T, TA<:AbstractMatrix, TB<:AbstractMatrix} <: AbstractKroneckerSum
    A::TA
    B::TB
    function KroneckerSum(A::AbstractMatrix{T}, B::AbstractMatrix{V}) where {T, V}
        (issquare(A) && issquare(B)) || throw(DimensionMismatch("KroneckerSum only applies to square matrices"))
        return new{promote_type(T, V), typeof(A), typeof(B)}(A, B)
    end
end

order(M::AbstractKroneckerSum) = order(M.A) + order(M.B)
issquare(M::AbstractKroneckerSum) = true

"""
    kroneckersum(A::AbstractMatrix, B::AbstractMatrix)

Construct a sum of Kronecker products between two square matrices and their
respective identity matrices. Does not evaluate the Kronecker products explicitly.
"""
kroneckersum(A::AbstractMatrix, B::AbstractMatrix) = KroneckerSum(A,B)

"""
    kroneckersum(A::AbstractMatrix, B::AbstractMatrix...)

Higher-order lazy kronecker sum, e.g.
```
kroneckersum(A,B,C,D)
```
"""
kroneckersum(A::AbstractMatrix, B::AbstractMatrix...) = kroneckersum(A,kroneckersum(B...))

"""
    kroneckersum(A::AbstractMatrix, pow::Int)

Kronecker-sum power, computes
`A ⊕ A ⊕ ... ⊕ A = (A ⊗ I ⊗ ... ⊗ I) + (I ⊗ A ⊗ ... ⊗ I) + ... (I ⊗ I ⊗ ... A)'.
"""
function kroneckersum(A::AbstractMatrix, pow::Int)
    @assert pow > 0 "Works only with positive powers!"
    if pow == 1
        return A
    else
        return A ⊕ kroneckersum(A, pow-1)
    end
end


"""
    ⊕(A::AbstractMatrix, B::AbstractMatrix)

Binary operator for `kroneckersum`, computes as Lazy Kronecker sum. See `kroneckersum` for
documentation.
"""
⊕(A::AbstractMatrix, B::AbstractMatrix) = kroneckersum(A, B)

⊕(A::AbstractMatrix...) = kroneckersum(A...)
⊕(A::AbstractMatrix, pow::Int) = kroneckersum(A, pow)

"""
    getmatrices(K::T) where T <: KroneckerSum

Obtain the two Kronecker products of a `KroneckerSum` object.
"""
Kronecker.getmatrices(K::AbstractKroneckerSum) = (K.A, K.B)

function Base.size(K::AbstractKroneckerSum)
    A, B = getmatrices(K)
    (m, n) = size(A)
    (k, l) = size(B)
    return m * k, n * l
end

Base.size(K::AbstractKroneckerSum, dim::Int) = size(K)[dim]

function Base.getindex(K::AbstractKroneckerSum, i1::Int, i2::Int)
    A, B = getmatrices(K)
    m, n = size(A)
    k, l = size(B)
    i₁, j₁ = cld(i1, k), cld(i2, l)
    i₂, j₂ = (i1 - 1) % k + 1, (i2 - 1) % l + 1
    v = zero(eltype(K))
    if i₁ == j₁
        v += B[i₂,j₂]
    end
    if i₂ == j₂
        v += A[i₁,j₁]
    end
    return v
end

function Base.eltype(K::AbstractKroneckerSum)
    A, B = getmatrices(K)
    return promote_type(eltype(A), eltype(B))
end

function LinearAlgebra.tr(K::AbstractKroneckerSum)
    A, B = getmatrices(K)
    n, m = size(A, 1), size(B, 1)
    return m * tr(A) + n * tr(B)
end


"""
    collect(K::AbstractKroneckerProduct)

Collects a lazy instance of the `AbstractKroneckerProduct` type into a full,
native matrix. Equivalent with `Matrix(K::AbstractKroneckerProduct)`.
"""
function Base.collect(K::AbstractKroneckerSum)
    A = Array{eltype(K)}(undef, size(K))
    for (ind, k) in enumerate(K)
        @inbounds A[ind] = k
    end
    return A
end


function Base.adjoint(K::AbstractKroneckerSum)
    A, B = getmatrices(K)
    return kroneckersum(A', B')
end

function Base.transpose(K::AbstractKroneckerSum)
    A, B = getmatrices(K)
    return kroneckersum(transpose(A),transpose(B))
end

function Base.conj(K::AbstractKroneckerSum)
    A, B = getmatrices(K)
    return kroneckersum(conj(A), conj(B))
end

function Base.exp(K::AbstractKroneckerSum)
    A, B = getmatrices(K)
    return kronecker(exp(A), exp(B))
end

#=
function Base.:*(K1::AbstractKroneckerSum, K2::AbstractKroneckerSum)

    # Collect products (not matrices)
    A, B = (K1.A, K1.B)
    C, D = (K2.A, K2.B)

    size.(getmatrices(A)) == size.(getmatrices(C)) || throw(DimensionMismatch("Mismatch between A and C in (A ⊗ B)(C ⊗ D)"))
    size.(getmatrices(B)) == size.(getmatrices(D)) || throw(DimensionMismatch("Mismatch between B and D in (A ⊗ B)(C ⊗ D)"))

    # Dimensions are also checked in src/base.jl
    return A*C + A*D + B*C + B*D
end
=#

function LinearAlgebra.mul!(x::AbstractVector, K::AbstractKroneckerSum, v::AbstractVector)
    A, B = getmatrices(K)
    a, b = size(A)
    c, d = size(B)
    e = length(v)
    f = length(x)
    f == a * c || throw(DimensionMismatch("Dimension missmatch between kronecker system and result placeholder"))
    e == b * d || throw(DimensionMismatch("Dimension missmatch between kronecker system and vector"))

    V = reshape(v, d, b)
    x .= vec(V * transpose(A)) .+ vec(B * V)
    return x
end

function Base.:*(K::AbstractKroneckerSum, v::AbstractVector)
    return mul!(Vector{promote_type(eltype(v), eltype(K))}(undef, first(size(K))), K, v)
end
