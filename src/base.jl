abstract type GeneralizedKroneckerProduct <: AbstractMatrix{Real} end

abstract type AbstractKroneckerProduct <: GeneralizedKroneckerProduct end

Base.IndexStyle(::Type{<:GeneralizedKroneckerProduct}) = IndexCartesian()

# general Kronecker product between two matrices
struct KroneckerProduct{T,TA<:AbstractMatrix, TB<:AbstractMatrix} <: AbstractKroneckerProduct
    A::TA
    B::TB
    function KroneckerProduct(A::AbstractMatrix{T}, B::AbstractMatrix{V}) where {T, V}
        return new{promote_type(T, V), typeof(A), typeof(B)}(A, B)
    end
end

"""
    issquare(A::AbstractMatrix)

Checks if an array is a square matrix.
"""
function issquare(A::AbstractMatrix)
    m, n = size(A)
    return m == n
end

"""
    issquare(K::AbstractKroneckerProduct)

Checks if all matrices of a Kronecker product are square.
"""
issquare(K::AbstractKroneckerProduct) = issquare(K.A) && issquare(K.B)

squarecheck(K::AbstractKroneckerProduct) = issquare(K) || throw(
            DimensionMismatch(
                "kronecker system is not composed of two square matrices: " *
                                        "$size(K.A) and $size(K.B)"))

"""
    issymmetric(K::AbstractKroneckerProduct)

Checks if a Kronecker product is symmetric.
"""
function LinearAlgebra.issymmetric(K::AbstractKroneckerProduct)
    return squarecheck(K) && issymmetric(K.A) && issymmetric(K.B)
end

"""
    order(M::AbstractMatrix)

Returns the order of a matrix, i.e. how many matrices are involved in the
Kronecker product (default to 1 for general matrices).
"""
order(M::AbstractMatrix) = 1
order(M::AbstractKroneckerProduct) = order(M.A) + order(M.B)

"""
    kronecker(A::AbstractMatrix, B::AbstractMatrix)

Construct a Kronecker product object between two arrays. Does not evaluate the
Kronecker product explictly.
"""
kronecker(A::AbstractMatrix, B::AbstractMatrix) = KroneckerProduct(A, B)

"""
    kronecker(A::AbstractMatrix, B::AbstractMatrix)

Higher-order Kronecker lazy kronecker product, e.g.
```
kronecker(A, B, C, D)
```
"""
kronecker(A::AbstractMatrix, B::AbstractMatrix...) = kronecker(A,
                                                            kronecker(B...))

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

Binary operator for `kronecker`, computes as Lazy Kronecker product. See
`kronecker` for documentation.
"""
⊗(A::AbstractMatrix, B::AbstractMatrix) = kronecker(A, B)
⊗(A::AbstractMatrix...) = kronecker(A...)
⊗(A::AbstractMatrix, pow::Int) = kronecker(A, pow)

"""
    getmatrices(K::AbstractKroneckerProduct)

Obtain the two matrices of an `AbstractKroneckerProduct` object.
"""
getmatrices(K::AbstractKroneckerProduct) = (K.A, K.B)

"""
    getmatrices(A::AbstractArray)

Returns a matrix itself. Needed for recursion.
"""
getmatrices(A::AbstractArray) = (A,)

function Base.size(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    (m, n) = size(A)
    (k, l) = size(B)
    return m * k, n * l
end

Base.size(K::GeneralizedKroneckerProduct, dim::Int) = size(K)[dim]

function Base.getindex(K::AbstractKroneckerProduct, i1::Int, i2::Int)
    A, B = getmatrices(K)
    m, n = size(A)
    k, l = size(B)
    return A[cld(i1, k), cld(i2, l)] * B[(i1 - 1) % k + 1, (i2 - 1) % l + 1]
end

function Base.eltype(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return promote_type(eltype(A), eltype(B))
end

"""
    tr(K::AbstractKroneckerProduct)

Compute the trace of a Kronecker product.
"""
function LinearAlgebra.tr(K::AbstractKroneckerProduct)
    squarecheck(K)
    return tr(K.A) * tr(K.B)
end

function LinearAlgebra.det(K::AbstractKroneckerProduct)
    squarecheck(K)
    A, B = getmatrices(K)
    m = size(A)[1]
    n = size(B)[1]
    return det(K.A)^n * det(K.B)^m
end

function LinearAlgebra.logdet(K::AbstractKroneckerProduct)
    squarecheck(K)
    A, B = getmatrices(K)
    m = size(A)[1]
    n = size(B)[1]
    return n * logdet(A) + m * logdet(B)
end

function Base.inv(K::AbstractKroneckerProduct)
    squarecheck(K)
    A, B = getmatrices(K)
    return KroneckerProduct(inv(A), inv(B))
end

"""
    collect(K::AbstractKroneckerProduct)

Collects a lazy instance of the `AbstractKroneckerProduct` type into a full,
native matrix. Equivalent with `Matrix(K::AbstractKroneckerProduct)`.
"""
function collect(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kron(A, B)
end

"""
    Matrix(K::GeneralizedKroneckerProduct)

Converts a `GeneralizedKroneckerProduct` instance to a Matrix type.
"""
Base.Matrix(K::GeneralizedKroneckerProduct) = collect(K)

function Base.adjoint(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kronecker(A', B')
end

function Base.transpose(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kronecker(transpose(A), transpose(B))
end

function Base.conj(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kronecker(conj(A), conj(B))
end

"""
    isposdef(K::AbstractKroneckerProduct)

Test whether a Kronecker product is positive definite (and Hermitian) by trying
to perform a Cholesky factorization of K.
"""
function LinearAlgebra.isposdef(K::AbstractKroneckerProduct)
    squarecheck(K)
    return isposdef(K.A) && isposdef(K.B)
end

function Base.kron(K::AbstractKroneckerProduct, C::AbstractMatrix)
    A, B = getmatrices(K)
    return kron(kron(A, B), C)
end

function Base.kron(A::AbstractMatrix, K::AbstractKroneckerProduct)
    B, C = getmatrices(K)
    return kron(A, kron(B, C))
end

Base.kron(K1::AbstractKroneckerProduct,
            K2::AbstractKroneckerProduct) = kron(collect(K1), collect(K2))

# mixed-product property
function Base.:*(K1::AbstractKroneckerProduct, K2::AbstractKroneckerProduct)
    A, B = getmatrices(K1)
    C, D = getmatrices(K2)
    # check for size
    if size(A, 2) != size(C, 1)
        throw(DimensionMismatch("Mismatch between A and C in (A ⊗ B)(C ⊗ D)"))
    end
    if size(B, 2) != size(D, 1)
        throw(DimensionMismatch("Mismatch between B and D in (A ⊗ B)(C ⊗ D)"))
    end
    return (A * C) ⊗ (B * D)
end

"""
    mul!(x::AbstractVector, K::AbstractKroneckerProduct, v::AbstractVector)

Calculates the vector-matrix multiplication `K * v` and stores the result in
`x`, overwriting its existing value.
"""
function mul!(x::AbstractVector, K::AbstractKroneckerProduct, v::AbstractVector)
    M, N = getmatrices(K)
    a, b = size(M)
    c, d = size(N)
    e = length(v)
    f = length(x)
    f == a * c || throw(DimensionMismatch(
        "Dimension missmatch between kronecker system and result placeholder"))
    e == b * d || throw(DimensionMismatch(
        "Dimension missmatch between kronecker system and vector"))
    if (d + a) * b < (b + c) * d
        x .= vec(N * (reshape(v, d, b) * transpose(M)))
    else
        x .= vec((N * reshape(v, d, b)) * transpose(M))
    end
    return x
end

"""
    mul!(x::AbstractVector, K::AbstractKroneckerProduct, v::AbstractVector)

Calculates the vector-matrix multiplication `K * v` and stores the result in
`x`, overwriting its existing value.
"""
function mul!(x::AbstractVector, K::AbstractKroneckerProduct, v::AbstractVector)
    M, N = getmatrices(K)
    a, b = size(M)
    c, d = size(N)
    e = length(v)
    f = length(x)
    f == a * c || throw(DimensionMismatch(
        "Dimension missmatch between kronecker system and result placeholder"))
    e == b * d || throw(DimensionMismatch(
        "Dimension missmatch between kronecker system and vector"))
    if (d + a) * b < (b + c) * d
        x .= vec(N * (reshape(v, d, b) * transpose(M)))
    else
        x .= vec((N * reshape(v, d, b)) * transpose(M))
    end
    return x
end

import Base.ReshapedArray

function mul!(x::AbstractVector, K::AbstractKroneckerProduct,
                                            v::ReshapedArray)
    M, N = getmatrices(K)
    a, b = size(M)
    c, d = size(N)
    e = length(v)
    f = length(x)
    f == a * c || throw(DimensionMismatch(
        "Dimension missmatch between kronecker system and result placeholder"))
    e == b * d || throw(DimensionMismatch(
        "Dimension missmatch between kronecker system and vector"))

    if size(v.parent) == (d, b)
        V = v.parent
    else
        V = reshape(v, d, b)
    end
    if (d + a) * b < (b + c) * d
        x .= vec(N * (V * transpose(M)))
    else
        x .= vec((N * V) * transpose(M))
    end
    return x
end


function Base.:*(K::AbstractKroneckerProduct, v::AbstractVector)
    return mul!(Vector{promote_type(eltype(v), eltype(K))}(undef,
                                                        first(size(K))), K, v)
end
