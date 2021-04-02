function ChainRulesCore.frule((_, ΔA, ΔB), ::typeof(KroneckerProduct), A::AbstractMatrix, B::AbstractMatrix)
    Ω = (A ⊗ B)
    ∂Ω = (ΔA ⊗ B) + (A ⊗ ΔB)
    return Ω, ∂Ω
end

function ChainRulesCore.rrule(::typeof(KroneckerProduct), A::AbstractMatrix, B::AbstractMatrix)
    function kronecker_product_pullback(ΔΩ)
        nA = size(A, 2)
        IA_col = Diagonal(ones(nA))
        ∂A = ΔΩ * (IA_col ⊗ B')
        
        nB = size(B, 2)
        IB_col = Diagonal(ones(nB))
        ∂B = ΔΩ * (A' ⊗ IB_col)
        return (NO_FIELDS, ∂A, ∂B)
    end
    return (A ⊗ B), kronecker_product_pullback
end