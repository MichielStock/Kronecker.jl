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
