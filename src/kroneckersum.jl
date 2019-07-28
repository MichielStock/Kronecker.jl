struct KroneckerSum{T<:SquareKroneckerProduct, S<:SquareKroneckerProduct}
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



order(M::KroneckerSum) = order(M.A) + order(M.B)

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
