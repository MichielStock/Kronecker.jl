abstract type AbstractKroneckerSum <: GeneralizedKroneckerProduct end

struct KroneckerSum{T<:SquareKroneckerProduct, S<:SquareKroneckerProduct} <: AbstractKroneckerSum
    # Fields
    A::T # (A ⊗ I_B)
    B::S # (I_A ⊗ B)

    function KroneckerSum(A::AbstractMatrix{T}, B::AbstractMatrix{V}) where {T, V}
        (issquare(A) && issquare(B)) || throw(DimensionMismatch("KroneckerSum only applies to square matrices"))
        AI = A ⊗ Diagonal(oneunit(B))
        IB = Diagonal(oneunit(A)) ⊗ B

        return new{typeof(AI),typeof(IB)}(AI, IB)
    end
end


# All products are of same order
order(M::AbstractKroneckerSum) = order(M.A)

"""
    kroneckersum(A::AbstractMatrix, B::AbstractMatrix)

Construct a sum of Kronecker products between two square matrices and their respective identity matrices.
Does not evaluate the Kronecker products explicitly.
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

⊕(A, B) = kroneckersum(A, B)

"""
    getmatrices(K::T) where T <: KroneckerSum

Obtain the two Kronecker products of a `KroneckerSum` object.
"""
Kronecker.getmatrices(K::AbstractKroneckerSum) = (Kronecker.getmatrices(K.A)[1], Kronecker.getmatrices(K.B)[2])

function Base.size(K::AbstractKroneckerSum)
    # All products in sum are the same size
    size(K.A)
end

function Base.getindex(K::AbstractKroneckerSum, i1::Int, i2::Int)
    A, B = (K.A, K.B)
    m, n = size(A)
    k, l = size(B)
    return getindex(A,i1,i2) + getindex(B,i1,i2)
end


Base.size(K::AbstractKroneckerSum, dim::Int) = size(K)[dim]

function Base.eltype(K::AbstractKroneckerSum)
    A, B = getmatrices(K)
    return promote_type(eltype(A), eltype(B))
end

# NOTE that K.A and K.B are both SquareKroneckerProducts, not the factor matrices
LinearAlgebra.tr(K::AbstractKroneckerSum) = tr(K.A) + tr(K.B)


function Base.collect(K::AbstractKroneckerSum)
    return K.A + K.B
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

function Base.:*(K1::AbstractKroneckerSum, K2::AbstractKroneckerSum)

    # Collect products (not matrices)
    A, B = (K1.A, K1.B)
    C, D = (K2.A, K2.B)

    size.(getmatrices(A)) == size.(getmatrices(C)) || throw(DimensionMismatch("Mismatch between A and C in (A ⊗ B)(C ⊗ D)"))
    size.(getmatrices(B)) == size.(getmatrices(D)) || throw(DimensionMismatch("Mismatch between B and D in (A ⊗ B)(C ⊗ D)"))

    # Dimensions are also checked in src/base.jl
    return A*C + A*D + B*C + B*D
end


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
