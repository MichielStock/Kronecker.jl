abstract type GeneralizedKroneckerProduct
end

abstract type KroneckerProduct <: GeneralizedKroneckerProduct
end

# TODO: make diagonal Kronecker system
# TODO: document functions
# TODO: indexing in general
# QUESTION: remove issquare checks?

# QUESTION: allow for different types of matrices?

struct KroneckerProductArray{T} <: KroneckerProduct where T <: AbstractArray
    A::T
    B::T
end

"""
    issquare(A::Array{T,2}) where T <: Real

Checks if an array is a square matrix.
"""
function issquare(A::Array{T,2}) where T <: Real
    m, n = size(A)
    return m == n
end

"""
    kronecker(A::T, B::T) where T <: AbstractArray

Construct a Kronecker product object between two arrays. Does not evaluate the
Kronecker product explictly!
"""
function kronecker(A::T, B::T) where T <: AbstractArray
    return KroneckerProductArray(A, B)
end

"""
    ⊗(A::T, B::T) where T <: AbstractArray

Construct a Kronecker product object between two arrays. Does not evaluate the
Kronecker product explictly!
"""
function ⊗(A::T, B::T) where T <: AbstractArray
    return KroneckerProductArray(A, B)
end

"""
    getmatrices(K::T) where T <: KroneckerProduct

Obtain the two matrices of a `KroneckerPoduct` object.
"""
function getmatrices(K::T) where T <: KroneckerProduct
    A = K.A
    B = K.B
    return A, B
end

"""
     Base.:size(K::T) where T <: KroneckerProductArray

Get the size of a `KroneckerPoduct` object.
"""
function Base.:size(K::T) where T <: KroneckerProductArray
    A, B = getmatrices(K)
    (m, n) = size(A)
    (k, l) = size(B)
    return m * k, n * l
end

"""
     Base.:size(K::T) where T <: KroneckerProductArray

Get the size of a `KroneckerPoduct` object.
"""
function Base.:size(K::T, dim::I where I<:Int) where T <: GeneralizedKroneckerProduct
    return size(K)[dim]
end

function Base.:show(io::IO, K::T) where T <: KroneckerProductArray
    A, B = getmatrices(K)
    print(io, "A ⊗ B")
end

function Base.:eltype(K::T) where T <: KroneckerProductArray
    A, B = getmatrices(K)
    return promote_type(eltype(A), eltype(B))
end

function LinearAlgebra.:tr(K::KroneckerProduct)
    (issquare(K.A) & issquare(K.B)) || throw(DimensionMismatch("Both matrices have to be square"))
    return tr(K.A) * tr(K.B)
end

function LinearAlgebra.:det(K::KroneckerProduct)
    A, B = getmatrices(K)
    (issquare(A) & issquare(B)) || throw(DimensionMismatch("Both matrices have to be square"))
    m = size(A)[1]
    n = size(B)[1]
    return det(K.A)^n * det(K.B)^m
end

function Base.:inv(K::T) where T <: KroneckerProduct
    A, B = getmatrices(K)
    return inv(A) ⊗ inv(B)
end

function Base.:collect(K::T) where T <: KroneckerProduct
    A, B = getmatrices(K)
    return kron(A, B)
end

function Base.:adjoint(K::T) where T <: KroneckerProduct
    A, B = getmatrices(K)
    return A' ⊗ B'
end

# mixed-product property
function Base.:*(K1::T where T <: KroneckerProduct,
                    K2::T where T <: KroneckerProduct)
    A, B = getmatrices(K1)
    C, D = getmatrices(K2)
    # check for size
    size(A, 2) == size(C, 1) || throw(DimensionMismatch("Mismatch between A and C in (A ⊗ B)(C ⊗ D)"))
    size(B, 2) == size(D, 1) || throw(DimensionMismatch("Mismatch between B and D in (A ⊗ B)(C ⊗ D)"))
    return (A * C) ⊗ (B * D)
end

function Base.:getindex(K::T, i1::Int64, i2::Int64) where T <: KroneckerProduct
    A, B = getmatrices(K)
    m, n = size(A)
    k, l = size(B)
    return A[cld(i1, k), cld(i2, l)] * B[(i1 - 1) % k + 1, (i2 - 1) % l + 1]
end

function mult!(x::V, K::T where T <: KroneckerProduct,
                v::V) where V <: AbstractVector{R} where R <: Real
    M, N = getmatrices(K)
    a, b = size(M)
    c, d = size(N)
    e = length(v)
    f = length(x)
    f == a * c || throw(DimensionMismatch("Dimension missmatch between kronecker system and result placeholder"))
    e == b * d || throw(DimensionMismatch("Dimension missmatch between kronecker system and vector"))
    if (d + a) * b < (b + c) * d
        x[:] .= vec(N * (reshape(v, d, b) * M'))
    else
        x[:] .= vec((N * reshape(v, d, b)) * M')
    end
    return x
end

function Base.:*(K::T where T <: KroneckerProduct,
                    v::V where V <: AbstractVector{R} where R <: Real)
    ac, bd = size(K)
    x = typeof(v)(undef, ac)
    return mult!(x, K, v)
end
