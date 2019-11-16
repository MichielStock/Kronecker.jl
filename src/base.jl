abstract type GeneralizedKroneckerProduct{T} <: AbstractMatrix{T} end

Base.eltype(K::GeneralizedKroneckerProduct{T}) where {T} = T

abstract type AbstractKroneckerProduct{T} <: GeneralizedKroneckerProduct{T} end

Base.IndexStyle(::Type{<:GeneralizedKroneckerProduct}) = IndexCartesian()

"""
    KroneckerProduct{T,TA<:AbstractMatrix, TB<:AbstractMatrix} <: AbstractKroneckerProduct{T}

Concrete Kronecker product between two matrices `A` and `B`.
"""
struct KroneckerProduct{T<:Any,TA<:AbstractMatrix, TB<:AbstractMatrix} <: AbstractKroneckerProduct{T}
    A::TA
    B::TB
    function KroneckerProduct(A::AbstractMatrix{T}, B::AbstractMatrix{V}) where {T, V}
        return new{promote_type(T, V), typeof(A), typeof(B)}(A, B)
    end
end

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
    ⊗(A::AbstractMatrix, B::AbstractMatrix)

Binary operator for `kronecker`, computes as Lazy Kronecker product. See
`kronecker` for documentation.
"""
⊗(A::AbstractMatrix, B::AbstractMatrix) = kronecker(A, B)
⊗(A::AbstractMatrix...) = kronecker(A...)

"""
    getindex(K::AbstractKroneckerProduct, i1::Integer, i2::Integer)

Computes and returns the (i,j)-th element of an `AbstractKroneckerProduct` K.
Uses recursion if `K` is of an order greater than two.
"""
function getindex(K::AbstractKroneckerProduct, i1::Integer, i2::Integer)
    A, B = getmatrices(K)
    m, n = size(A)
    k, l = size(B)
    return A[cld(i1, k), cld(i2, l)] * B[(i1 - 1) % k + 1, (i2 - 1) % l + 1]
end

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

"""
    size(K::AbstractKroneckerProduct)

Returns a the size of an `AbstractKroneckerProduct` instance.
"""
function size(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    (m, n) = size(A)
    (k, l) = size(B)
    return m * k, n * l
end

"""
    size(K::GeneralizedKroneckerProduct)

Returns a the size of an `GeneralizedKroneckerProduct` instance.
"""
size(K::GeneralizedKroneckerProduct, dim::Integer) = size(K)[dim]

# CHECKS

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
    isposdef(K::AbstractKroneckerProduct)

Test whether a Kronecker product is positive definite (and Hermitian) by trying
to perform a Cholesky factorization of K.
"""
function LinearAlgebra.isposdef(K::AbstractKroneckerProduct)
    squarecheck(K)
    return isposdef(K.A) && isposdef(K.B)
end

# SCALAR PROPERTIES

"""
    order(M::AbstractMatrix)

Returns the order of a matrix, i.e. how many matrices are involved in the
Kronecker product (default to 1 for general matrices).
"""
order(M::AbstractMatrix) = 1
order(M::AbstractKroneckerProduct) = order(M.A) + order(M.B)

"""
    det(K::AbstractKroneckerProduct)

Compute the determinant of a Kronecker product.
"""
function LinearAlgebra.det(K::AbstractKroneckerProduct)
    squarecheck(K)
    A, B = getmatrices(K)
    m = size(A)[1]
    n = size(B)[1]
    return det(K.A)^n * det(K.B)^m
end

"""
    logdet(K::AbstractKroneckerProduct)

Compute the logarithm of the determinant of a Kronecker product.
"""
function LinearAlgebra.logdet(K::AbstractKroneckerProduct)
    squarecheck(K)
    A, B = getmatrices(K)
    m = size(A)[1]
    n = size(B)[1]
    return n * logdet(A) + m * logdet(B)
end

"""
    tr(K::AbstractKroneckerProduct)

Compute the trace of a Kronecker product.
"""
function LinearAlgebra.tr(K::AbstractKroneckerProduct)
    squarecheck(K)
    return tr(K.A) * tr(K.B)
end

"""
    inv(K::AbstractKroneckerProduct)

Compute the inverse of a Kronecker product.
"""
function inv(K::AbstractKroneckerProduct)
    squarecheck(K)
    A, B = getmatrices(K)
    return KroneckerProduct(inv(A), inv(B))
end

"""
    adjoint(K::AbstractKroneckerProduct)

Compute the adjoint of a Kronecker product.
"""
function adjoint(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kronecker(A', B')
end

"""
    transpose(K::AbstractKroneckerProduct)

Compute the transpose of a Kronecker product.
"""
function Base.transpose(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kronecker(transpose(A), transpose(B))
end

function Base.conj(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kronecker(conj(A), conj(B))
end

# COLLECTING

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

function Base.:+(A::AbstractKroneckerProduct, B::StridedMatrix)
    C = Matrix(A)
    C .+= B
    return C
end
Base.:+(A::StridedMatrix, B::AbstractKroneckerProduct) = B + A

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
    lmul!(a::Number, K::AbstractKroneckerProduct)

Scale an `AbstractKroneckerProduct` `K` inplace by a factor `a` by rescaling the
**left** matrix.
"""
function LinearAlgebra.lmul!(a::Number, K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    lmul!(a, A)
end

"""
    rmul!(K::AbstractKroneckerProduct, a::Number)

Scale an `AbstractKroneckerProduct` `K` inplace by a factor `a` by rescaling the
**right** matrix.
"""
function LinearAlgebra.rmul!(K::AbstractKroneckerProduct, a::Number)
    A, B = getmatrices(K)
    rmul!(B, a)
end

function Base.:*(a::Number, K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    kronecker(a * A, B)
end


function Base.:*(K::AbstractKroneckerProduct, a::Number)
    A, B = getmatrices(K)
    kronecker(A, B * a)
end

# SOLVING
function LinearAlgebra.:\(K::AbstractKroneckerProduct{T}, c::AbstractVector{T}) where {T}
    size(K, 1) != length(c) && throw(DimensionMismatch("size(K, 1) != length(c)"))
    C = reshape(c, size(K.B, 1), size(K.A, 1)) # matricify
    return vec((K.B \ C) / K.A') #(A ⊗ B)vec(X) = vec(C) <=> BXA' = C => X = B^{-1} C A'^{-1}
end
