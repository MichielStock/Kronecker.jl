abstract type GeneralizedKroneckerProduct <: AbstractMatrix{Real}
end

Base.IndexStyle(::Type{<:GeneralizedKroneckerProduct}) = IndexLinear()

# TODO: make diagonal Kronecker system
# TODO: document functions
# TODO: indexing in general
# QUESTION: remove issquare checks?

# QUESTION: allow for different types of matrices?

# general Kronecker product between two matrices
struct KroneckerProduct <: GeneralizedKroneckerProduct
    A::AbstractMatrix
    B::AbstractMatrix
end

"""
    issquare(A::AbstractMatrix) where T <: Real

Checks if an array is a square matrix.
"""
function issquare(A::AbstractMatrix)
    m, n = size(A)
    return m == n
end

# general Kronecker product between two matrices
struct SquareKroneckerProduct <: GeneralizedKroneckerProduct
    A::AbstractMatrix
    B::AbstractMatrix
    function SquareKroneckerProduct(A, B)
        if issquare(A) && issquare(B)
            new(A, B)
        else
            throw(DimensionMismatch("SquareKroneckerProduct is only for when all matrices are square"))
        end
    end
end

issquare(K::SquareKroneckerProduct) = true

KronProd = Union{KroneckerProduct, SquareKroneckerProduct}

"""
    kronecker(A::AbstractMatrix, B::AbstractMatrix)

Construct a Kronecker product object between two arrays. Does not evaluate the
Kronecker product explictly!
"""
function kronecker(A::AbstractMatrix, B::AbstractMatrix)
    if issquare(A) && issquare(B)
        return SquareKroneckerProduct(A, B)
    else
        return KroneckerProduct(A, B)
    end
end

"""
    ⊗(A::AbstractMatrix, B::AbstractMatrix)

Construct a Kronecker product object between two arrays. Does not evaluate the
Kronecker product explictly!
"""
function ⊗(A::AbstractMatrix, B::AbstractMatrix)
    return kronecker(A, B)
end

"""
    getmatrices(K::T) where T <: KroneckerProduct

Obtain the two matrices of a `KroneckerPoduct` object.
"""
function getmatrices(K::T) where T <: KronProd
    A = K.A
    B = K.B
    return A, B
end

"""
     Base.:size(K::T) where T <: KroneckerProduct

Get the size of a `KroneckerPoduct` object.
"""
function Base.:size(K::T) where T <: KronProd
    A, B = getmatrices(K)
    (m, n) = size(A)
    (k, l) = size(B)
    return m * k, n * l
end

function Base.:getindex(K::KronProd, i1::Int, i2::Int)
    A, B = getmatrices(K)
    m, n = size(A)
    k, l = size(B)
    return A[cld(i1, k), cld(i2, l)] * B[(i1 - 1) % k + 1, (i2 - 1) % l + 1]
end

"""
     Base.:size(K::T) where T <: KroneckerProduct

Get the size of a `KroneckerPoduct` object.
"""
function Base.:size(K::GeneralizedKroneckerProduct, dim::I where I<:Int)
    return size(K)[dim]
end

function Base.:eltype(K::T) where T <: KronProd
    A, B = getmatrices(K)
    return promote_type(eltype(A), eltype(B))
end

function LinearAlgebra.:tr(K::SquareKroneckerProduct)
    return tr(K.A) * tr(K.B)
end

function LinearAlgebra.:det(K::SquareKroneckerProduct)
    A, B = getmatrices(K)
    m = size(A)[1]
    n = size(B)[1]
    return det(K.A)^n * det(K.B)^m
end

function Base.:inv(K::SquareKroneckerProduct)
    A, B = getmatrices(K)
    return SquareKroneckerProduct(inv(A), inv(B))
end

function Base.:collect(K::T) where T <: KronProd
    A, B = getmatrices(K)
    return kron(A, B)
end

function Base.:adjoint(K::T) where T <: KronProd
    A, B = getmatrices(K)
    return kronecker(A', B')
end

# mixed-product property
function Base.:*(K1::KronProd,
                    K2::KronProd)
    A, B = getmatrices(K1)
    C, D = getmatrices(K2)
    # check for size
    size(A, 2) == size(C, 1) || throw(DimensionMismatch("Mismatch between A and C in (A ⊗ B)(C ⊗ D)"))
    size(B, 2) == size(D, 1) || throw(DimensionMismatch("Mismatch between B and D in (A ⊗ B)(C ⊗ D)"))
    return (A * C) ⊗ (B * D)
end

function mult!(x::V, K::T where T <: KronProd,
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

function Base.:*(K::T where T <: KronProd,
                    v::V where V <: AbstractVector{R} where R <: Real)
    ac, bd = size(K)
    x = typeof(v)(undef, ac)
    return mult!(x, K, v)
end
