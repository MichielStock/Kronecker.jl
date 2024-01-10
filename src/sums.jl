#=
Created on 03/01/2021 11:18:20
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Sums of matrices (including a Kronecker type)
=#

"""
    SumOfKroneckers{T<:Any, TA<:GeneralizedKroneckerProduct,
                            TB<:AbstractMatrix} <: GeneralizedKroneckerProduct{T}

Sums of Kroneckers is a lazy sum of Kronecker terms (at least the first term has to be of the `GeneralizedKroneckerProduct` type).
Its main use is to fascilitate the distributive property. All matrix properties that can be computed lazily are supported (such as
`sum` and `adjoint`) while nonlinear operations are not (e.g., `inv`, `eigen`, `det`).
"""
struct SumOfKroneckers{T<:Any, TA<:GeneralizedKroneckerProduct,
                            TB<:AbstractMatrix} <: GeneralizedKroneckerProduct{T}
    KA::TA
    KB::TB
    function SumOfKroneckers(KA::GeneralizedKroneckerProduct{T}, KB::AbstractMatrix{V}) where {T, V}
        return new{promote_type(T, V), typeof(KA), typeof(KB)}(KA, KB)
    end
end

Base.size(K::SumOfKroneckers) = size(K.KA)

Base.getindex(K::SumOfKroneckers, i1::Integer, i2::Integer) = K.KA[i1,i2] + K.KB[i1,i2] 

function Base.:+(KA::GeneralizedKroneckerProduct, KB::AbstractMatrix)
    @assert size(KA) == size(KB) "`A` and `B` have to be conformable"
    return SumOfKroneckers(KA, KB)
end

Base.collect(K::SumOfKroneckers) = collect(K.KA) .+ collect(K.KB)

Base.:*(K::SumOfKroneckers, V::VecOrMat) = K.KA * V .+ K.KB * V

# linear operations
LinearAlgebra.tr(K::SumOfKroneckers) = tr(K.KA) + tr(K.KB)
Base.adjoint(K::SumOfKroneckers) = adjoint(K.KA) + adjoint(K.KB)
Base.transpose(K::SumOfKroneckers) = transpose(K.KA) + transpose(K.KB)
Base.conj(K::SumOfKroneckers) = conj(K.KA) + conj(K.KB)

Base.:*(K::SumOfKroneckers, a::Number) = K.KA * a + K.KB * a
Base.:*(a::Number, K::SumOfKroneckers) = a * K.KA + a * K.KB


function Base.:*(K::KroneckerProduct{<:Any,<:SumOfKroneckers,<:AbstractMatrix}, V::VecOrMat)
    A, B = getmatrices(K)
    return (A.KA ⊗ B) * V .+ ((A.KB ⊗ B) * V)
end

function Base.:*(K::KroneckerProduct{<:Any,<:AbstractMatrix,<:SumOfKroneckers}, V::VecOrMat)
    A, B = getmatrices(K)
    return (A ⊗ B.KA) * V .+ (A ⊗ B.KB) * V
end

function Base.:*(K::KroneckerProduct{<:Any,<:SumOfKroneckers,<:SumOfKroneckers}, V::VecOrMat)
    A, B = getmatrices(K)
    return (A.KA ⊗ B.KA) * V .+ (A.KB ⊗ B.KB) * V .+ (A.KB ⊗ B.KA) * V .+ (A.KB ⊗ B.KB) * V 
end

function Base.sum(K::SumOfKroneckers; dims::Union{Int,Nothing}=nothing)
    isnothing(dims) && return sum(K.KA) + sum(K.KB)
    return sum(K.KA, dims=dims) + sum(K.KB, dims=dims)
end


