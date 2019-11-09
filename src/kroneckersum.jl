abstract type AbstractKroneckerSum{T} <: GeneralizedKroneckerProduct{T} end

struct KroneckerSum{T<:Any, TA<:AbstractMatrix, TB<:AbstractMatrix} <: AbstractKroneckerSum{T}
    A::TA
    B::TB
    function KroneckerSum(A::AbstractMatrix{T},
                            B::AbstractMatrix{V}) where {T, V}
        (issquare(A) && issquare(B)) || throw(DimensionMismatch(
                                "KroneckerSum only applies to square matrices"))
        return new{promote_type(T, V), typeof(A), typeof(B)}(A, B)
    end
end

order(M::AbstractKroneckerSum) = order(M.A) + order(M.B)
issquare(M::AbstractKroneckerSum) = true

"""
    kroneckersum(A::AbstractMatrix, B::AbstractMatrix)

Construct a sum of Kronecker products between two square matrices and their
respective identity matrices. Does not evaluate the Kronecker products
explicitly.
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
`A ⊕ A ⊕ ... ⊕ A = (A ⊗ I ⊗ ... ⊗ I) + (I ⊗ A ⊗ ... ⊗ I) + ... (I ⊗ I ⊗ ... A)`.
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
    getmatrices(K::T) where T <: AbstractKroneckerSum

Obtain the two matrices of an `AbstractKroneckerSum` object.
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
    collect(K::AbstractKroneckerSum)

Collects a lazy instance of the `AbstractKroneckerSum` type into a full,
native matrix. Returns the result as a sparse matrix.
"""
function Base.collect(K::AbstractKroneckerSum)
    A, B = getmatrices(K)
    A, B = sparse(A), sparse(B)
    IA, IB = oneunit(A), oneunit(B)
    return kron(A, IB) + kron(IA, B)
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

"""
    exp(K::AbstractKroneckerSum)

Computes the matrix exponential of an `AbstractKroneckerSum` `K`. Returns an
instance of `KroneckerProduct`.
"""
function Base.exp(K::AbstractKroneckerSum)
    A, B = getmatrices(K)
    return kronecker(exp(A), exp(B))
end

#=
function Base.sum(K::KroneckerSum)
    A, B = getmatrices(K)
    n, m = size(A, 1), size(B, 1)
    return m * sum(A) + n * sum(B)
end
=#

function Base.sum(K::KroneckerSum; dims::Union{Int,Nothing}=nothing)
    A, B = getmatrices(K)
    n, m = size(A, 1), size(B, 1)
    if dims==1
        return kron(sum(A, dims=1), ones(1, m)) .+ kron(ones(1, n), sum(B, dims=1))
    elseif dims==2
        return kron(sum(A, dims=2), ones(m, 1)) .+ kron(ones(n, 1), sum(B, dims=2))
    elseif dims==nothing
        m * sum(A) + n * sum(B)
    else
        throw(ArgumentError("`dims` should be 1 or 2"))
    end
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

if VERSION < v"1.3.0-alpha.115"
function LinearAlgebra.mul!(x::AbstractVector, K::AbstractKroneckerSum, v::AbstractVector)
    A, B = getmatrices(K)
    a, b = size(A)
    c, d = size(B)
    e = length(v)
    f = length(x)
    f == a * c || throw(DimensionMismatch("Dimension missmatch between kronecker system and result placeholder"))
    e == b * d || throw(DimensionMismatch("Dimension missmatch between kronecker system and vector"))

    V = reshape(v, d, b)
    X = reshape(x, c, a)
    mul!(X, V, transpose(A))
    X .+= B * V
    return x
end
else # 5-arg mul! is available
function LinearAlgebra.mul!(x::AbstractVector, K::AbstractKroneckerSum, v::AbstractVector)
    A, B = getmatrices(K)
    a, b = size(A)
    c, d = size(B)
    e = length(v)
    f = length(x)
    f == a * c || throw(DimensionMismatch("Dimension missmatch between kronecker system and result placeholder"))
    e == b * d || throw(DimensionMismatch("Dimension missmatch between kronecker system and vector"))

    V = reshape(v, d, b)
    X = reshape(x, c, a)
    mul!(X, V, transpose(A))
    mul!(X, B, V, true, true)
    return x
end
end # VERSION

function Base.:*(K::AbstractKroneckerSum, v::AbstractVector)
    return mul!(Vector{promote_type(eltype(v), eltype(K))}(undef, first(size(K))), K, v)
end
