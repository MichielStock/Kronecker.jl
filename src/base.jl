abstract type GeneralizedKroneckerProduct{T} <: AbstractMatrix{T} end

Base.eltype(K::GeneralizedKroneckerProduct{T}) where {T} = T

abstract type AbstractKroneckerProduct{T} <: GeneralizedKroneckerProduct{T} end

Base.IndexStyle(::Type{<:GeneralizedKroneckerProduct}) = IndexCartesian()

"""
    collect!(C::AbstractMatrix, K::GeneralizedKroneckerProduct)

In-place collection of `K` in `C`. If possible, specialized routines are used to
speed up the computation. The fallback is an element-wise iteration. In this case,
this function might be slow.
"""
function collect!(C::AbstractMatrix, K::GeneralizedKroneckerProduct)
    size(C) == size(K) || throw(DimensionMismatch("`K` $(size(K)) cannot be collected in `C` $(size(C))"))
    @inbounds for I in CartesianIndices(K)
        C[I] = K[I]
    end
    return C
end


"""
    collect(K::GeneralizedKroneckerProduct)

Collects a lazy instance of the `GeneralizedKroneckerProduct` type into a dense,
native matrix. Falls back to the element-wise case when not specialized method
is defined.
"""
function collect(K::GeneralizedKroneckerProduct{T}) where {T}
    C = Matrix{T}(undef, size(K)...)
    return collect!(C, K)
end


"""
    KroneckerProduct{T,TA<:AbstractMatrix, TB<:AbstractMatrix} <: AbstractKroneckerProduct{T}

Concrete Kronecker product between two matrices `A` and `B`.
"""
struct KroneckerProduct{T<:Any, TA<:AbstractMatrix, TB<:AbstractMatrix} <: AbstractKroneckerProduct{T}
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
    getallmatrices(K::AbstractKroneckerProduct)

Obtain all matrices in an `AbstractKroneckerProduct` object.
"""
getallmatrices(K::AbstractKroneckerProduct) = (getallmatrices(K.A)..., getallmatrices(K.B)...)
getallmatrices(K::AbstractMatrix) = (K,)


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
    issymmetric(K::AbstractKroneckerProduct)

Checks if a Kronecker product is symmetric.
"""
function LinearAlgebra.issymmetric(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return issymmetric(A) && issymmetric(B)
end


"""
    ishermitian(K::AbstractKroneckerProduct)

Checks if a Kronecker product is Hermitian.
"""
function LinearAlgebra.ishermitian(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return ishermitian(A) && ishermitian(B)
end

"""
    isposdef(K::AbstractKroneckerProduct)

Test whether a Kronecker product is positive definite (and Hermitian) by trying
to perform a Cholesky factorization of K.
"""
function LinearAlgebra.isposdef(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return isposdef(A) && isposdef(B)
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
function LinearAlgebra.det(K::AbstractKroneckerProduct{T}) where {T}
    checksquare(K)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        m = size(A)[1]
        n = size(B)[1]
        return det(K.A)^n * det(K.B)^m
    else
        return zero(T)
    end
end

"""
    logdet(K::AbstractKroneckerProduct)

Compute the logarithm of the determinant of a Kronecker product.
"""
function LinearAlgebra.logdet(K::AbstractKroneckerProduct{T}) where {T}
    checksquare(K)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        m = size(A)[1]
        n = size(B)[1]
        return n * logdet(A) + m * logdet(B)
    else
        return real(T)(-Inf)
    end
end

"""
    tr(K::AbstractKroneckerProduct)

Compute the trace of a Kronecker product.
"""
function LinearAlgebra.tr(K::AbstractKroneckerProduct)
    checksquare(K)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        return tr(A) * tr(B)
    else
        return sum(diag(K))
    end
end

"""
    inv(K::AbstractKroneckerProduct)

Compute the inverse of a Kronecker product.
"""
function inv(K::AbstractKroneckerProduct)
    checksquare(K)
    A, B = getmatrices(K)
    if issquare(A) && issquare(B)
        return KroneckerProduct(inv(A), inv(B))
    else
        throw(SingularException(1))
    end
end

"""
    pinv(K::AbstractKroneckerProduct)

Compute the Moore-Penrose pseudo-inverse of a Kronecker product.
"""
function LinearAlgebra.pinv(K::AbstractKroneckerProduct)
    checksquare(K)
    A, B = getmatrices(K)
    return KroneckerProduct(pinv(A), pinv(B))
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

#=
"""
    collect(K::AbstractKroneckerProduct)

Collects a lazy instance of the `AbstractKroneckerProduct` type into a full,
native matrix. Equivalent with `Matrix(K::AbstractKroneckerProduct)`.
"""
function collect(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return kron(A, B)
end
=#


# function for in-place Kronecker product
function _kron!(C::AbstractArray, A::AbstractArray, B::AbstractArray)
    m = 0
    @inbounds for j = 1:size(A,2), l = 1:size(B,2), i = 1:size(A,1)
        Aij = A[i,j]
        for k = 1:size(B,1)
            C[m += 1] = Aij * B[k,l]
        end
    end
    return C
end

_kron!(C::AbstractArray, A::GeneralizedKroneckerProduct, B::AbstractArray) = _kron!(C, collect(A), B)
_kron!(C::AbstractArray, A::AbstractArray, B::GeneralizedKroneckerProduct) = _kron!(C, A, collect(B))
_kron!(C::AbstractArray, A::GeneralizedKroneckerProduct, B::GeneralizedKroneckerProduct) = _kron!(C, collect(A), collect(B))

"""
    collect!(C::AbstractMatrix, K::AbstractKroneckerProduct)

In-place collection of `K` in `C` where `K` is an `AbstractKroneckerProduct`, i.e.,
`K = A ⊗ B`.
"""
function collect!(C::AbstractMatrix, K::AbstractKroneckerProduct)
    size(C) == size(K) || throw(DimensionMismatch("`K` $(size(K)) cannot be collected in `C` $(size(C))"))
    A, B = getmatrices(K)
    return _kron!(C, A, B)
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
