abstract type GeneralizedKroneckerProduct <: AbstractMatrix{Real}
end

abstract type AbstractKroneckerProduct <: GeneralizedKroneckerProduct
end

"""
    Matrix(K::GeneralizedKroneckerProduct)

Converts a `GeneralizedKroneckerProduct` instance to a Matrix type.
"""
Matrix(K::GeneralizedKroneckerProduct) = collect(K)

Base.IndexStyle(::Type{<:GeneralizedKroneckerProduct}) = IndexLinear()

# general Kronecker product between two matrices
struct KroneckerProduct <: AbstractKroneckerProduct
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
struct SquareKroneckerProduct <: AbstractKroneckerProduct
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
LinearAlgebra.:issymmetric(K::SquareKroneckerProduct) = issymmetric(K.A) && issymmetric(K.B)

"""
    order(M::AbstractMatrix)

Returns the order of a matrix, i.e. how many matrices are
involved in the Kronecker product (default to 1 for general
matrices).
"""
order(M::AbstractMatrix) = 1
order(M::AbstractKroneckerProduct) = order(M.A) + order(M.B)

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
    kronecker(A::AbstractMatrix, B::AbstractMatrix)

Higher-order Kronecker lazy kronecker product, e.g.
```
kronecker(A, B, C, D)
```
"""
kronecker(A::AbstractMatrix, B::AbstractMatrix...) = kronecker(A, kronecker(B...))

"""
    kronecker(A::AbstractMatrix, pow::Int)

Kronecker power, computes `A ⊗ A ⊗ ... ⊗ A`.
"""
function kronecker(A::AbstractMatrix, pow::Int)
    @assert pow > 0 "Works only with positive powers!"
    if pow == 1
        return A
    else
        return A ⊗ kronecker(A, pow-1)
    end
end


"""
    ⊗(A::AbstractMatrix, B::AbstractMatrix)

Binary operator for `kronecker`, computes as Lazy Kronecker
product. See `kronecker` for documentation.
"""
function ⊗(A::AbstractMatrix, B::AbstractMatrix)
    return kronecker(A, B)
end

⊗(A, B) = kronecker(A, B)

"""
    getmatrices(K::T) where T <: KroneckerProduct

Obtain the two matrices of a `KroneckerPoduct` object.
"""
function getmatrices(K::AbstractKroneckerProduct)
    A = K.A
    B = K.B
    return A, B
end

"""
     Base.:size(K::T) where T <: KroneckerProduct

Get the size of a `KroneckerPoduct` object.
"""
function Base.:size(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    (m, n) = size(A)
    (k, l) = size(B)
    return m * k, n * l
end

function Base.:getindex(K::AbstractKroneckerProduct, i1::Int, i2::Int)
    A, B = getmatrices(K)
    m, n = size(A)
    k, l = size(B)
    return A[cld(i1, k), cld(i2, l)] * B[(i1 - 1) % k + 1, (i2 - 1) % l + 1]
end

"""
     Base.:size(K::T) where T <: KroneckerProduct

Get the size of a `KroneckerPoduct` object.
"""
function Base.:size(K::GeneralizedKroneckerProduct, dim::Int)
    return size(K)[dim]
end

function Base.:eltype(K::AbstractKroneckerProduct)
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

function Base.:collect(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kron(A, B)
end

function Base.:adjoint(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kronecker(A', B')
end

function Base.:transpose(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kronecker(transpose(A), transpose(B))
end

# mixed-product property
function Base.:*(K1::AbstractKroneckerProduct,
                    K2::AbstractKroneckerProduct)
    A, B = getmatrices(K1)
    C, D = getmatrices(K2)
    # check for size
    size(A, 2) == size(C, 1) || throw(DimensionMismatch("Mismatch between A and C in (A ⊗ B)(C ⊗ D)"))
    size(B, 2) == size(D, 1) || throw(DimensionMismatch("Mismatch between B and D in (A ⊗ B)(C ⊗ D)"))
    return (A * C) ⊗ (B * D)
end

function mult!(x::AbstractVector, K::AbstractKroneckerProduct,
                v::AbstractVector)
    M, N = getmatrices(K)
    a, b = size(M)
    c, d = size(N)
    e = length(v)
    f = length(x)
    f == a * c || throw(DimensionMismatch("Dimension missmatch between kronecker system and result placeholder"))
    e == b * d || throw(DimensionMismatch("Dimension missmatch between kronecker system and vector"))
    if (d + a) * b < (b + c) * d
        x .= vec(N * (reshape(v, d, b) * transpose(M)))
    else
        x .= vec((N * reshape(v, d, b)) * transpose(M))
    end
    return x
end

#=
function mult!(x::AbstractVector{Complex}, K::AbstractKroneckerProduct,
                v::AbstractVector)
    M, N = getmatrices(K)
    a, b = size(M)
    c, d = size(N)
    e = length(v)
    f = length(x)
    f == a * c || throw(DimensionMismatch("Dimension missmatch between kronecker system and result placeholder"))
    e == b * d || throw(DimensionMismatch("Dimension missmatch between kronecker system and vector"))
    if (d + a) * b < (b + c) * d
        x .= vec(N * (reshape(v, d, b) * transpose(M)))
    else
        x .= vec((N * reshape(v, d, b)) * transpose(M))
    end
    return x
end
=#

function Base.:*(K::T where T <: AbstractKroneckerProduct,
                    v::V where V <: AbstractVector)
    ac, bd = size(K)
    x = Vector{promote_type(eltype(v), eltype(K))}(undef, ac)
    return mult!(x, K, v)
end
