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
